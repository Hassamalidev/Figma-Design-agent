"""Deterministic scaffolding: palette parsing and the scripts we generate.

Pure logic -- no Figma, no model. The generated JS is compiled the same way
the plugin evaluates it, so a syntax slip fails here rather than mid-run.
"""
from __future__ import annotations

import re
import subprocess

import pytest

from agent import scaffold


def compiles_as_async_body(code: str) -> bool:
    """The plugin runs each script as `new AsyncFunction(code)` -- do the same."""
    result = subprocess.run(
        ["node", "-e", "new (Object.getPrototypeOf(async function(){}).constructor)(process.argv[1])", "--", code],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_palette_is_read_from_the_models_own_brief():
    brief = """
    - `--color-primary: #0066FF` (accent)
    - `--color-bg: #FFFFFF`
    - Text Dark: #222222
    """
    palette = scaffold.extract_palette(brief)

    names = [n for n, _ in palette]
    hexes = [h for _, h in palette]
    assert "#0066FF" in hexes and "#FFFFFF" in hexes and "#222222" in hexes
    assert "primary" in names  # the "--color-" prefix is stripped
    assert "text-dark" in names


def test_duplicate_colours_are_collapsed():
    brief = "primary: #0066FF\nalso-primary: #0066FF\nbg: #FFFFFF"

    palette = scaffold.extract_palette(brief)

    assert [h for _, h in palette] == ["#0066FF", "#FFFFFF"]


def test_a_brief_with_no_colours_falls_back_to_a_real_palette():
    palette = scaffold.extract_palette("Make it clean and modern with generous spacing.")

    assert palette == scaffold.FALLBACK_PALETTE
    assert all(re.fullmatch(r"#[0-9A-F]{6}", h) for _, h in palette)


def test_hex_conversion_is_0_to_1_not_0_to_255():
    assert scaffold.hex_to_rgb("#FFFFFF") == (1.0, 1.0, 1.0)
    assert scaffold.hex_to_rgb("#000000") == (0.0, 0.0, 0.0)
    r, g, b = scaffold.hex_to_rgb("#0066FF")
    assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1
    assert b == 1.0


def test_token_script_uses_only_real_api_calls():
    """The exact calls live runs hallucinated must not appear here."""
    script = scaffold.build_token_script([("accent", "#0066FF")])

    assert "createVariableCollection" in script
    assert "setBoundVariableForPaint" in script
    assert "getLocalVariableCollectionsAsync" in script
    for invented in ("createVariableSet", "getVariableByName", "createPaintStyleAsync", "?."):
        assert invented not in script


def test_scripts_return_the_expected_shape():
    assert "createdNodeIds" in scaffold.build_token_script([("a", "#000000")])
    assert "createdNodeIds" in scaffold.build_text_style_script()
    assert "createdNodeIds" in scaffold.build_placeholder_section_script("1:2", "Footer")


def test_text_styles_use_valid_inter_style_strings():
    script = scaffold.build_text_style_script()

    assert "Semi Bold" in script  # with a space -- "SemiBold" fails to load
    assert "SemiBold" not in script.replace("Semi Bold", "")
    assert "'PIXELS'" in script or '"PIXELS"' in script  # lineHeight is an object


def test_placeholder_appends_into_the_root_and_is_obvious():
    script = scaffold.build_placeholder_section_script("42:7", "Pricing table")

    assert '"42:7"' in script
    assert "appendChild" in script
    assert "TODO" in script
    # Sizing comes after appendChild, or FILL would be rejected.
    assert script.index("appendChild") < script.index("layoutSizingHorizontal")


def test_bind_script_only_rebinds_near_matches():
    """Rebinding a colour the model clearly meant as a token is a fix; changing
    a deliberate one-off colour would be a design decision, so it isn't done."""
    script = scaffold.build_bind_fills_script(["1:2"], [("accent", "#0066FF")])

    assert "setFillStyleIdAsync" in script
    assert "stillUnbound" in script          # non-matches are reported, not changed
    assert str(scaffold.BIND_TOLERANCE) in script
    assert "fillStyleId" in script           # already-bound nodes are skipped
    # Tolerance must be tight enough to be visually invisible.
    assert scaffold.BIND_TOLERANCE <= 0.1


def test_bind_script_carries_the_real_palette_values():
    script = scaffold.build_bind_fills_script(["1:2"], [("accent", "#0066FF")])

    assert "color/accent" in script
    assert "0.4" in script  # 0x66 / 255, so the nearest-match maths is real


def test_hug_fix_targets_the_root_and_reports_whether_it_changed():
    script = scaffold.build_hug_fix_script("9:1")

    assert '"9:1"' in script
    assert "primaryAxisSizingMode" in script
    assert "changed" in script


@pytest.mark.parametrize(
    "script",
    [
        scaffold.build_token_script(scaffold.FALLBACK_PALETTE),
        scaffold.build_text_style_script(),
        scaffold.build_placeholder_section_script("1:2", 'Weird "quoted" label'),
        scaffold.build_bind_fills_script(["1:2", "3:4"], scaffold.FALLBACK_PALETTE),
        scaffold.build_hug_fix_script("9:1"),
    ],
)
def test_generated_scripts_are_valid_javascript(script):
    assert compiles_as_async_body(script)


# ---- palette roles and contrast -------------------------------------------

# Deep navy, warm sand, coral accent -- the shape a real brief produces.
PALETTE = [("deep-navy", "#0B1F3A"), ("warm-sand", "#E8DCC8"), ("coral", "#FF6B4A")]


def test_contrast_ratio_matches_wcag_reference_values():
    """Black on white is exactly 21:1; a colour against itself is 1:1."""
    assert scaffold.contrast_ratio("#000000", "#FFFFFF") == 21.0
    assert scaffold.contrast_ratio("#FFFFFF", "#FFFFFF") == 1.0


def test_luminance_orders_colours_light_to_dark():
    assert scaffold.relative_luminance("#FFFFFF") > scaffold.relative_luminance("#E8DCC8")
    assert scaffold.relative_luminance("#E8DCC8") > scaffold.relative_luminance("#0B1F3A")


def test_roles_are_derived_from_luminance_not_from_the_name():
    roles = {name: role for name, _, role in scaffold.describe_palette(PALETTE)}
    assert roles["color/warm-sand"].startswith("background")
    assert roles["color/deep-navy"].startswith("text")
    assert roles["color/coral"].startswith("accent")


def test_token_names_are_preserved_so_style_lookups_still_work():
    """The paint styles were created under these names -- renaming would break
    every `s.name === 'color/...'` lookup in a generated script.
    """
    names = [name for name, _, _ in scaffold.describe_palette(PALETTE)]
    assert names == ["color/deep-navy", "color/warm-sand", "color/coral"]


def test_pairings_only_include_combinations_that_actually_pass_aa():
    pairings = scaffold.readable_pairings(PALETTE)
    assert any("color/deep-navy text on color/warm-sand background" in p for p in pairings)
    # Coral on warm sand is ~2:1 -- unreadable, and must not be offered.
    assert not any("coral text on color/warm-sand" in p for p in pairings)
    assert all(float(p.split("(")[1].split(":")[0]) >= scaffold.MIN_TEXT_CONTRAST for p in pairings)


def test_a_palette_with_no_readable_pair_returns_nothing():
    """Two near-identical greys cannot make readable text; say so with silence
    rather than offering an illegal pairing."""
    assert scaffold.readable_pairings([("a", "#777777"), ("b", "#808080")]) == []


def test_describe_palette_tolerates_an_empty_palette():
    assert scaffold.describe_palette([]) == []
    assert scaffold.readable_pairings([]) == []


def test_a_neutral_only_palette_has_no_accent():
    """Nothing saturated enough to be a brand colour -- don't invent one."""
    roles = [role for _, _, role in scaffold.describe_palette(
        [("white", "#FFFFFF"), ("grey", "#888888"), ("black", "#111111")]
    )]
    assert not any(r.startswith("accent") for r in roles)


def test_a_mid_dark_neutral_reads_as_muted_text_not_a_card_surface():
    """Slate is the second-lightest neutral, but it is dark -- calling it a
    card surface would put dark cards on a light page."""
    roles = {n: r for n, _, r in scaffold.describe_palette(
        PALETTE + [("slate", "#5A6B7C")]
    )}
    assert roles["color/slate"].startswith("text-muted")


def test_a_light_mid_neutral_does_read_as_a_surface():
    roles = {n: r for n, _, r in scaffold.describe_palette(
        [("white", "#FFFFFF"), ("off-white", "#F2F4F7"), ("ink", "#101828")]
    )}
    assert roles["color/off-white"].startswith("surface")


# ---- a real trace: one duplicate name destroyed the whole palette ----------

DASHBOARD_BRIEF = """
  - Accent: Orange - #FF6600.
  - Text: Dark Gray - #333333.
  - Card background: White (#FFFFFF).
  - Status badges: Green (#28A745) for Delivered, Orange (#FFC107) for In Transit.
"""


def test_two_colours_with_the_same_name_are_disambiguated():
    """A live dashboard run had "Accent: Orange #FF6600" and "Orange (#FFC107)".
    Figma rejects a duplicate variable name, and because scripts are atomic that
    single collision discarded EVERY token -- the run built with no styles at all.
    """
    palette = scaffold.extract_palette(DASHBOARD_BRIEF)
    names = [name for name, _ in palette]
    assert len(names) == len(set(names)), f"duplicate token name in {names}"
    assert "orange" in names and "orange-2" in names
    # Both colours are kept -- #FFC107 is a real status colour, not a mistake.
    hexes = {h for _, h in palette}
    assert "#FF6600" in hexes and "#FFC107" in hexes


def test_the_token_script_reports_a_bad_entry_instead_of_losing_the_palette():
    """One malformed colour must cost one token, never all of them."""
    script = scaffold.build_token_script([("ok", "#112233"), ("also-ok", "#445566")])
    assert "try {" in script and "catch" in script
    assert "failedTokens" in script


def test_the_token_script_tracks_names_it_creates_during_the_loop():
    """It re-scanned a snapshot taken before the loop, so a name created inside
    the loop was invisible and got created twice."""
    script = scaffold.build_token_script([("a", "#112233")])
    assert "varByName[varName] = variable" in script
    assert "styleByName[varName] = style" in script


# ---- a real trace: nine requested colours became three tokens --------------
#
# The user asked for a nine-colour palette in a clean list. The run created
# THREE tokens, because the palette was only ever read out of the model's
# rewritten brief -- which had scattered the same colours into table cells.
# `border` and `surface` then aliased onto the page fill, so every divider was
# white on white and the input boxes had no visible edge.

NEXORA_INSTRUCTION = """
### Color Palette
* Deep background: `#0B1020`
* Primary purple: `#7C3AED`
* Secondary blue: `#2563EB`
* Accent: `#A78BFA`
* Form background: `#FFFFFF`
* Main text: `#111827`
* Secondary text: `#6B7280`
* Border: `#E5E7EB`
* Input background: `#F9FAFB`
"""

NEXORA_BRIEF = """
- Background: `#0B1020`.
- Gradient: Linear from `#7C3AED` to `#2563EB`.
| Input field | 48 px | 1 px solid `#E5E7EB` | `#F9FAFB` |
- Background: `#FFFFFF`.
"""


def test_the_users_instruction_is_read_before_the_models_brief():
    """Reading only the brief lost six of the nine colours that were asked for."""
    from_brief_only = scaffold.extract_palette(NEXORA_BRIEF)
    assert len(from_brief_only) < 4  # the bug, reproduced

    palette = scaffold.extract_palette(NEXORA_BRIEF, NEXORA_INSTRUCTION)

    hexes = {h for _, h in palette}
    for asked_for in ("#0B1020", "#7C3AED", "#2563EB", "#A78BFA",
                      "#FFFFFF", "#111827", "#6B7280", "#E5E7EB", "#F9FAFB"):
        assert asked_for in hexes, f"{asked_for} was requested and dropped"


def _roles_for(instruction: str) -> dict[str, str]:
    """role key -> hex, exactly as the renderer resolves it."""
    from agent import renderer

    palette = scaffold.complete_palette(scaffold.extract_palette("", instruction))
    info = scaffold.describe_palette(palette)
    hexes = {name: hex_value for name, hex_value, _ in info}
    return {role: hexes[token] for role, token in renderer.role_map(info).items()}


def test_a_name_that_declares_its_role_beats_luminance_ordering():
    """`Border: #E5E7EB` has told us the answer. Ranking it by luminance made it
    a "surface", made #111827 "text-muted", and called the near-black page
    background "text"."""
    roles = _roles_for(NEXORA_INSTRUCTION)

    assert roles["border"] == "#E5E7EB"
    assert roles["text"] == "#111827"
    assert roles["text-muted"] == "#6B7280"
    assert roles["surface"] == "#F9FAFB"


def test_the_background_role_is_one_the_text_can_be_read_on():
    """Two colours claim to be backgrounds. Picking the dark one puts near-black
    text on a near-black page -- every gate passes it and nothing is legible."""
    roles = _roles_for(NEXORA_INSTRUCTION)

    assert scaffold.contrast_ratio(roles["background"], roles["text"]) >= scaffold.MIN_TEXT_CONTRAST


def test_the_second_background_stays_available_as_an_inverse_panel():
    """A dark-to-light split screen needs the dark half. With only one
    `background` role it asked for the page fill, got white, and rendered a
    1440x900 blank void."""
    roles = _roles_for(NEXORA_INSTRUCTION)

    assert roles["background-alt"] == "#0B1020"
    # ...and copy placed on it is readable, measured rather than assumed.
    assert scaffold.contrast_ratio(
        roles["text-on-alt"], roles["background-alt"]
    ) >= scaffold.MIN_TEXT_CONTRAST


def test_a_missing_role_is_derived_rather_than_aliased_onto_the_page_fill():
    """The old chain was border -> surface -> background, so a two-colour
    palette painted dividers white on white."""
    palette = scaffold.complete_palette([("ink", "#111111"), ("paper", "#FFFFFF")])
    roles = {
        role: hex_value
        for role, hex_value in (
            (r.split(" ")[0], h) for _, h, r in scaffold.describe_palette(palette)
        )
    }

    assert roles["border"] != roles["background"]
    assert roles["surface"] != roles["border"]
    assert scaffold.contrast_ratio(roles["border"], roles["background"]) > 1.0


def test_completing_a_palette_that_needs_nothing_changes_nothing():
    complete = scaffold.complete_palette(PALETTE)
    assert complete[: len(PALETTE)] == PALETTE
