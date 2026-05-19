# FR3 robot assets (M1)

Bundled for offline `ModelBuilder.add_usd` import (no Omniverse `https://` references in the sim layer).

| Path | Role |
|------|------|
| **`../testfr3_resolved.usda`** | **Sim entrypoint** --- composes Isaac ``fr3`` from `omniverse_fr3/` plus your **testfr3**-style EE/tcp (same content as editing `testfr3.usd` in Isaac, but with local refs fixed for Newton) |
| **`../testfr3.usd`** | Authoring/export from Isaac Sim (may still point at Omniverse until you repoint; see below) |
| `omniverse_fr3/fr3.usd` | Isaac FR3 arm payload |
| `omniverse_fr3/configuration/fr3_robot_schema.usd` | Required sublayer for `fr3.usd` |

## Newton import

Use **`assets/testfr3_resolved.usda`** (constant `apple_pick_sim.fr3_robot.TESTFR3_SCENE_USD`). That file:

- Prepends `./fr3/omniverse_fr3/fr3.usd` instead of an Omniverse URL.
- Weld **EE → link7** (not link8), so rigid-body graph cycles are avoided when `tcp` is a non-dynamic Xform joint target.
- Adds **PhysicsRigidBodyAPI** + mass on **`/fr3/ee/tcp`** so tcp is imported as its own rigid link.

If you regenerate **`testfr3.usd`** from Isaac:

1. Replace the Omniverse **`fr3.usd`** reference with `@./fr3/omniverse_fr3/fr3.usd@` (relative to **`assets/`**), **or**
2. Keep using **`testfr3_resolved.usda`** and merge any EE deltas from your USD into its `def "fr3"` overlays.

Do not hand-edit binary `.usd` crates unless necessary; prefer `.usda` overlays like `testfr3_resolved.usda`.

## Regenerate Omniverse subtree (optional)

```bash
uv run --directory newton python -c "
from pathlib import Path
from newton._src.utils.import_usd import resolve_usd_from_url
root = Path('assets/fr3/omniverse_fr3').resolve()
resolve_usd_from_url(
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd',
    target_folder_name=str(root),
)
"
