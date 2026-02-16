# Blender MCP v2.0.0

**The ultimate Blender integration through the Model Context Protocol** — control everything in Blender via AI.

Created by **Aymen Mabrouk** | [MIT License](LICENSE)

---

## What is Blender MCP?

Blender MCP connects any AI assistant (via the Model Context Protocol) to Blender, giving the AI **full control** over your 3D workspace. The AI can create objects, apply materials, set up cameras and lighting, animate, render, and much more — all through natural language.

## Features

### 🎯 Direct Blender Tools (No Code Required)
| Category | Tools |
|----------|-------|
| **Object Creation** | `create_object` — Create any primitive (cube, sphere, cylinder, cone, torus, plane, monkey, text, curves, empties, cameras, lights) |
| **Object Management** | `delete_object`, `duplicate_object`, `set_transform`, `apply_transforms`, `join_objects`, `set_parent` |
| **Modifiers** | `add_modifier`, `remove_modifier`, `apply_modifier` — Supports all Blender modifiers (Subsurf, Mirror, Boolean, Array, Bevel, Solidify, etc.) |
| **Materials** | `create_material` (Principled BSDF with full PBR control), `assign_material` |
| **Camera** | `set_camera` — Position, rotate, set focal length, look-at target |
| **Lighting** | `add_light` — Point, Sun, Spot, Area lights with energy/color/size |
| **Animation** | `set_keyframe` — Location, rotation, scale keyframes at any frame |
| **Collections** | `manage_collection` — Create, list, move objects, delete collections |
| **Rendering** | `render_image` — Render with EEVEE, Cycles, or Workbench |
| **Export** | `export_scene` — Export to GLTF, GLB, FBX, OBJ, STL, PLY, USD, Alembic |
| **Scene Info** | `get_scene_info`, `get_object_info`, `get_viewport_screenshot` |
| **API Docs** | `search_blender_docs` — RAG search through complete Blender Python API documentation |

### 🐍 Execute Any Blender Python Code
The `execute_blender_code` tool lets the AI run **any Python code** inside Blender with full access to `bpy`, `mathutils`, `bmesh`, `math`, `os`, `json`, `Vector`, `Matrix`, `Euler`, `Quaternion`, `Color` — for mesh editing, node manipulation, physics, constraints, UV unwrapping, and anything else.

### 🌐 Asset Integrations
| Integration | Purpose |
|------------|---------|
| **Poly Haven** | Free HDRIs, textures, and 3D models |
| **Sketchfab** | Search and download realistic 3D models |
| **Hyper3D Rodin** | AI-generate custom 3D models from text or images |
| **Hunyuan3D** | Tencent's AI 3D model generation |

---

## Installation

### Prerequisites
- **Blender 3.0+** installed
- **Python 3.10+** with `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Step 1: Install the Blender Addon
1. Open Blender → Edit → Preferences → Add-ons
2. Click **Install** and select the `addon.py` file
3. Enable the **Blender MCP** addon

### Step 2: Configure Your AI Client
Add the following to your AI client's MCP configuration (e.g. Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "blender": {
      "command": "uvx",
      "args": ["blender-mcp"]
    }
  }
}
```

### Step 3: Connect
1. In Blender's 3D Viewport, open the sidebar (press **N**)
2. Find the **BlenderMCP** tab
3. Click **"Connect to MCP Server"**
4. Start chatting with your AI assistant!

---

## Optional Integrations

### Poly Haven (Free Assets)
- Check "Use assets from Poly Haven" in the BlenderMCP panel

### Sketchfab
- Check "Use assets from Sketchfab"
- Enter your [Sketchfab API key](https://sketchfab.com/settings/password)

### Hyper3D Rodin
- Check "Use Hyper3D Rodin 3D model generation"
- Use the free trial API key or get your own at [hyper3d.ai](https://hyper3d.ai)

### Hunyuan3D
- Check "Use Tencent Hunyuan 3D model generation"
- Choose Official API (requires Tencent Cloud credentials) or Local API

---

## Complete Tool List (25 Direct + 24 Integration = 49 Total)

### Core Tools (Always Available)
1. `ping_blender` — Fast connectivity check with scene snapshot
2. `get_mcp_capabilities` — Session capabilities snapshot (core + enabled integrations)
3. `get_scene_info` — Full scene metadata (50 objects, render settings, camera, frame range)
4. `get_object_info` — Detailed object data (modifiers, constraints, materials, children, bounding box)
5. `get_viewport_screenshot` — Visual verification
6. `execute_blender_code` — Run any Python code with full module access
7. `create_object` — Create primitives, cameras, lights, empties, curves, text
8. `delete_object` — Remove objects with optional children
9. `duplicate_object` — Full or linked duplicates
10. `set_transform` — Position, rotate, scale objects
11. `apply_transforms` — Freeze transforms
12. `join_objects` — Merge multiple meshes into one
13. `set_parent` — Set/clear parent-child relationships
14. `add_modifier` — Add any modifier with custom parameters
15. `remove_modifier` — Remove by name
16. `apply_modifier` — Apply permanently
17. `create_material` — Principled BSDF (base color, metallic, roughness, emission, alpha)
18. `assign_material` — Assign to object
19. `set_camera` — Configure with look-at, focal length
20. `add_light` — POINT, SUN, SPOT, AREA with full control
21. `set_keyframe` — Animation keyframes (location, rotation, scale, custom)
22. `manage_collection` — Create, list, move, delete collections
23. `render_image` — Render with any engine
24. `export_scene` — 8 export formats
25. `search_blender_docs` — Optional API reference lookup for advanced scripting

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BLENDER_HOST` | `localhost` | Blender connection host |
| `BLENDER_PORT` | `9876` | Blender connection port |
| `BLENDER_DOCS_PATH` | _(auto-discovered)_ | Absolute path to `blender_docs_md` for `search_blender_docs` (useful when addon is installed outside your project folder) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
