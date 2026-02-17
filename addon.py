# Blender MCP v2.0.0 - Created by Aymen Mabrouk

import re
import bpy
import mathutils
import json
import threading
import socket
import time
import requests
import tempfile
import traceback
import os
import shutil
import zipfile
from bpy.props import IntProperty, BoolProperty
import io
import html
from datetime import datetime
import hashlib, hmac, base64
import os.path as osp
from contextlib import redirect_stdout, suppress

bl_info = {
    "name": "Blender MCP",
    "author": "Aymen Mabrouk",
    "version": (2, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > BlenderMCP",
    "description": "The ultimate Blender integration through the Model Context Protocol",
    "category": "Interface",
}

RODIN_FREE_TRIAL_KEY = "k9TcfFoEhNd9cCPP2guHAHHHkctZHIRhZDywZ1euGUXwihbYLpOjQhofby80NJez"

# Add User-Agent as required by Poly Haven API
REQ_HEADERS = requests.utils.default_headers()
REQ_HEADERS.update({"User-Agent": "blender-mcp"})

POLYPIZZA_API_BASE = "https://poly.pizza/api/v1.1"

class BlenderMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.running = False
        self.socket = None
        self.server_thread = None

    def start(self):
        if self.running:
            print("Server is already running")
            return

        self.running = True

        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)

            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()

            print(f"BlenderMCP server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            self.stop()

    def stop(self):
        self.running = False

        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

        # Wait for thread to finish
        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None

        print("BlenderMCP server stopped")

    def _server_loop(self):
        """Main server loop in a separate thread"""
        print("Server thread started")
        self.socket.settimeout(1.0)  # Timeout to allow for stopping

        while self.running:
            try:
                # Accept new connection
                try:
                    client, address = self.socket.accept()
                    print(f"Connected to client: {address}")

                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    # Just check running condition
                    continue
                except Exception as e:
                    print(f"Error accepting connection: {str(e)}")
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error in server loop: {str(e)}")
                if not self.running:
                    break
                time.sleep(0.5)

        print("Server thread stopped")

    def _handle_client(self, client):
        """Handle connected client"""
        print("Client handler started")
        client.settimeout(None)  # No timeout
        buffer = b''

        try:
            while self.running:
                # Receive data
                try:
                    data = client.recv(8192)
                    if not data:
                        print("Client disconnected")
                        break

                    buffer += data
                    try:
                        # Try to parse command
                        command = json.loads(buffer.decode('utf-8'))
                        buffer = b''

                        # Execute command in Blender's main thread
                        def execute_wrapper():
                            try:
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                try:
                                    client.sendall(response_json.encode('utf-8'))
                                except:
                                    print("Failed to send response - client disconnected")
                            except Exception as e:
                                print(f"Error executing command: {str(e)}")
                                traceback.print_exc()
                                try:
                                    error_response = {
                                        "status": "error",
                                        "message": str(e)
                                    }
                                    client.sendall(json.dumps(error_response).encode('utf-8'))
                                except:
                                    pass
                            return None

                        # Schedule execution in main thread
                        bpy.app.timers.register(execute_wrapper, first_interval=0.0)
                    except json.JSONDecodeError:
                        # Incomplete data, wait for more
                        pass
                except Exception as e:
                    print(f"Error receiving data: {str(e)}")
                    break
        except Exception as e:
            print(f"Error in client handler: {str(e)}")
        finally:
            try:
                client.close()
            except:
                pass
            print("Client handler stopped")

    def execute_command(self, command):
        """Execute a command in the main Blender thread"""
        try:
            return self._execute_command_internal(command)

        except Exception as e:
            print(f"Error executing command: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _execute_command_internal(self, command):
        """Internal command execution with proper context"""
        cmd_type = command.get("type")
        params = command.get("params", {})

        # Add a handler for checking PolyHaven status
        if cmd_type == "get_polyhaven_status":
            return {"status": "success", "result": self.get_polyhaven_status()}

        # Base handlers that are always available
        handlers = {
            "get_scene_info": self.get_scene_info,
            "get_object_info": self.get_object_info,
            "get_viewport_screenshot": self.get_viewport_screenshot,
            "execute_code": self.execute_code,
            "get_polyhaven_status": self.get_polyhaven_status,
            "get_hyper3d_status": self.get_hyper3d_status,
            "get_sketchfab_status": self.get_sketchfab_status,
            "get_polypizza_status": self.get_polypizza_status,
            "get_hunyuan3d_status": self.get_hunyuan3d_status,
            # New dedicated tools - always available
            "create_object": self.create_object,
            "delete_object": self.delete_object,
            "set_transform": self.set_transform,
            "apply_transforms": self.apply_transforms,
            "add_modifier": self.add_modifier,
            "remove_modifier": self.remove_modifier,
            "apply_modifier": self.apply_modifier,
            "create_material": self.create_material,
            "assign_material": self.assign_material,
            "render_image": self.render_image,
            "set_camera": self.set_camera,
            "add_light": self.add_light,
            "manage_collection": self.manage_collection,
            "set_keyframe": self.set_keyframe,
            "export_scene": self.export_scene,
            "duplicate_object": self.duplicate_object,
            "join_objects": self.join_objects,
            "set_parent": self.set_parent,
            "select_objects": self.select_objects,
            "frame_control": self.frame_control,
            "save_blend_file": self.save_blend_file,
            "open_blend_file": self.open_blend_file,
            "import_file": self.import_file,
            "search_blender_docs": self.search_blender_docs,
        }

        # Add Polyhaven handlers only if enabled
        if bpy.context.scene.blendermcp_use_polyhaven:
            polyhaven_handlers = {
                "get_polyhaven_categories": self.get_polyhaven_categories,
                "search_polyhaven_assets": self.search_polyhaven_assets,
                "download_polyhaven_asset": self.download_polyhaven_asset,
                "set_texture": self.set_texture,
            }
            handlers.update(polyhaven_handlers)

        # Add Hyper3d handlers only if enabled
        if bpy.context.scene.blendermcp_use_hyper3d:
            polyhaven_handlers = {
                "create_rodin_job": self.create_rodin_job,
                "poll_rodin_job_status": self.poll_rodin_job_status,
                "import_generated_asset": self.import_generated_asset,
            }
            handlers.update(polyhaven_handlers)

        # Add Sketchfab handlers only if enabled
        if bpy.context.scene.blendermcp_use_sketchfab:
            sketchfab_handlers = {
                "search_sketchfab_models": self.search_sketchfab_models,
                "get_sketchfab_model_preview": self.get_sketchfab_model_preview,
                "download_sketchfab_model": self.download_sketchfab_model,
                "get_sketchfab_model_license": self.get_sketchfab_model_license,
            }
            handlers.update(sketchfab_handlers)

        # Add Poly Pizza handlers only if enabled
        if bpy.context.scene.blendermcp_use_polypizza:
            polypizza_handlers = {
                "search_polypizza_models": self.search_polypizza_models,
                "download_polypizza_model": self.download_polypizza_model,
            }
            handlers.update(polypizza_handlers)
        
        # Add Hunyuan3d handlers only if enabled
        if bpy.context.scene.blendermcp_use_hunyuan3d:
            hunyuan_handlers = {
                "create_hunyuan_job": self.create_hunyuan_job,
                "poll_hunyuan_job_status": self.poll_hunyuan_job_status,
                "import_generated_asset_hunyuan": self.import_generated_asset_hunyuan
            }
            handlers.update(hunyuan_handlers)

        handler = handlers.get(cmd_type)
        if handler:
            try:
                print(f"Executing handler for {cmd_type}")
                result = handler(**params)
                print(f"Handler execution complete")
                return {"status": "success", "result": result}
            except Exception as e:
                print(f"Error in handler: {str(e)}")
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}



    def get_scene_info(self):
        """Get comprehensive information about the current Blender scene"""
        import math
        try:
            print("Getting scene info...")
            scene = bpy.context.scene
            scene_info = {
                "name": scene.name,
                "object_count": len(scene.objects),
                "objects": [],
                "materials_count": len(bpy.data.materials),
                "active_object": bpy.context.active_object.name if bpy.context.active_object else None,
                "selected_objects": [o.name for o in bpy.context.selected_objects][:20],
                "render_engine": scene.render.engine,
                "render_resolution": [scene.render.resolution_x, scene.render.resolution_y],
                "frame_current": scene.frame_current,
                "frame_range": [scene.frame_start, scene.frame_end],
                "camera": scene.camera.name if scene.camera else None,
                "world": bpy.context.scene.world.name if bpy.context.scene.world else None,
            }

            # Collect object information (up to 50 objects)
            for i, obj in enumerate(scene.objects):
                if i >= 50:
                    scene_info["_truncated"] = f"Showing 50 of {len(scene.objects)} objects"
                    break

                obj_info = {
                    "name": obj.name,
                    "type": obj.type,
                    "location": [round(float(x), 3) for x in obj.location],
                    "rotation_degrees": [round(math.degrees(r), 1) for r in obj.rotation_euler],
                    "scale": [round(float(x), 3) for x in obj.scale],
                    "visible": obj.visible_get(),
                }
                if obj.parent:
                    obj_info["parent"] = obj.parent.name
                if obj.modifiers:
                    obj_info["modifiers"] = [f"{m.name}({m.type})" for m in obj.modifiers]
                if obj.type == 'MESH' and obj.data:
                    obj_info["vertices"] = len(obj.data.vertices)
                    obj_info["faces"] = len(obj.data.polygons)
                if obj.material_slots:
                    obj_info["materials"] = [s.material.name for s in obj.material_slots if s.material]

                scene_info["objects"].append(obj_info)

            print(f"Scene info collected: {len(scene_info['objects'])} objects")
            return scene_info
        except Exception as e:
            print(f"Error in get_scene_info: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    @staticmethod
    def _get_aabb(obj):
        """ Returns the world-space axis-aligned bounding box (AABB) of an object. """
        if obj.type != 'MESH':
            raise TypeError("Object must be a mesh")

        # Get the bounding box corners in local space
        local_bbox_corners = [mathutils.Vector(corner) for corner in obj.bound_box]

        # Convert to world coordinates
        world_bbox_corners = [obj.matrix_world @ corner for corner in local_bbox_corners]

        # Compute axis-aligned min/max coordinates
        min_corner = mathutils.Vector(map(min, zip(*world_bbox_corners)))
        max_corner = mathutils.Vector(map(max, zip(*world_bbox_corners)))

        return [
            [*min_corner], [*max_corner]
        ]



    def get_object_info(self, name):
        """Get detailed information about a specific object"""
        import math
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")

        # Basic object info
        obj_info = {
            "name": obj.name,
            "type": obj.type,
            "location": [round(x, 4) for x in obj.location],
            "rotation_degrees": [round(math.degrees(r), 2) for r in obj.rotation_euler],
            "scale": [round(x, 4) for x in obj.scale],
            "dimensions": [round(x, 4) for x in obj.dimensions],
            "visible": obj.visible_get(),
            "materials": [],
            "parent": obj.parent.name if obj.parent else None,
            "children": [c.name for c in obj.children],
            "collections": [c.name for c in obj.users_collection],
        }

        if obj.type == "MESH":
            bounding_box = self._get_aabb(obj)
            obj_info["world_bounding_box"] = bounding_box

        # Add material slots
        for slot in obj.material_slots:
            if slot.material:
                obj_info["materials"].append(slot.material.name)

        # Add mesh data if applicable
        if obj.type == 'MESH' and obj.data:
            mesh = obj.data
            obj_info["mesh"] = {
                "vertices": len(mesh.vertices),
                "edges": len(mesh.edges),
                "polygons": len(mesh.polygons),
            }

        # Add modifiers
        obj_info["modifiers"] = []
        for mod in obj.modifiers:
            mod_info = {
                "name": mod.name,
                "type": mod.type,
                "show_viewport": mod.show_viewport,
                "show_render": mod.show_render,
            }
            obj_info["modifiers"].append(mod_info)

        # Add constraints
        obj_info["constraints"] = [
            {"name": c.name, "type": c.type, "enabled": c.enabled}
            for c in obj.constraints
        ]

        # Add light/camera specific data
        if obj.type == 'LIGHT' and obj.data:
            obj_info["light"] = {
                "type": obj.data.type,
                "energy": obj.data.energy,
                "color": list(obj.data.color),
            }
        elif obj.type == 'CAMERA' and obj.data:
            obj_info["camera"] = {
                "lens": obj.data.lens,
                "clip_start": obj.data.clip_start,
                "clip_end": obj.data.clip_end,
                "type": obj.data.type,
            }

        return obj_info

    def get_viewport_screenshot(self, max_size=800, filepath=None, format="png"):
        """
        Capture a screenshot of the current 3D viewport and save it to the specified path.

        Parameters:
        - max_size: Maximum size in pixels for the largest dimension of the image
        - filepath: Path where to save the screenshot file
        - format: Image format (png, jpg, etc.)

        Returns success/error status
        """
        try:
            if not filepath:
                return {"error": "No filepath provided"}

            # Find the active 3D viewport
            area = None
            for a in bpy.context.screen.areas:
                if a.type == 'VIEW_3D':
                    area = a
                    break

            if not area:
                return {"error": "No 3D viewport found"}

            # Take screenshot with proper context override
            with bpy.context.temp_override(area=area):
                bpy.ops.screen.screenshot_area(filepath=filepath)

            # Load and resize if needed
            img = bpy.data.images.load(filepath)
            width, height = img.size

            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img.scale(new_width, new_height)

                # Set format and save
                img.file_format = format.upper()
                img.save()
                width, height = new_width, new_height

            # Cleanup Blender image data
            bpy.data.images.remove(img)

            return {
                "success": True,
                "width": width,
                "height": height,
                "filepath": filepath
            }

        except Exception as e:
            return {"error": str(e)}

    def execute_code(self, code):
        """Execute arbitrary Blender Python code with full module access"""
        try:
            import math
            import bmesh

            # Rich namespace with all essential modules for Blender scripting
            namespace = {
                "__builtins__": __builtins__,
                "bpy": bpy,
                "mathutils": mathutils,
                "math": math,
                "bmesh": bmesh,
                "os": os,
                "json": json,
                "re": re,
                # Common mathutils shortcuts
                "Vector": mathutils.Vector,
                "Matrix": mathutils.Matrix,
                "Euler": mathutils.Euler,
                "Quaternion": mathutils.Quaternion,
                "Color": mathutils.Color,
            }

            # Capture stdout during execution, and return it as result
            capture_buffer = io.StringIO()
            with redirect_stdout(capture_buffer):
                exec(code, namespace)

            captured_output = capture_buffer.getvalue()
            return {"executed": True, "result": captured_output}
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            raise Exception(f"Code execution error: {error_msg}\n{tb}")



    def get_polyhaven_categories(self, asset_type):
        """Get categories for a specific asset type from Polyhaven"""
        try:
            if asset_type not in ["hdris", "textures", "models", "all"]:
                return {"error": f"Invalid asset type: {asset_type}. Must be one of: hdris, textures, models, all"}

            response = requests.get(f"https://api.polyhaven.com/categories/{asset_type}", headers=REQ_HEADERS)
            if response.status_code == 200:
                return {"categories": response.json()}
            else:
                return {"error": f"API request failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def search_polyhaven_assets(self, asset_type=None, categories=None):
        """Search for assets from Polyhaven with optional filtering"""
        try:
            url = "https://api.polyhaven.com/assets"
            params = {}

            if asset_type and asset_type != "all":
                if asset_type not in ["hdris", "textures", "models"]:
                    return {"error": f"Invalid asset type: {asset_type}. Must be one of: hdris, textures, models, all"}
                params["type"] = asset_type

            if categories:
                params["categories"] = categories

            response = requests.get(url, params=params, headers=REQ_HEADERS)
            if response.status_code == 200:
                # Limit the response size to avoid overwhelming Blender
                assets = response.json()
                # Return only the first 20 assets to keep response size manageable
                limited_assets = {}
                for i, (key, value) in enumerate(assets.items()):
                    if i >= 20:  # Limit to 20 assets
                        break
                    limited_assets[key] = value

                return {"assets": limited_assets, "total_count": len(assets), "returned_count": len(limited_assets)}
            else:
                return {"error": f"API request failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def download_polyhaven_asset(self, asset_id, asset_type, resolution="1k", file_format=None):
        try:
            # First get the files information
            files_response = requests.get(f"https://api.polyhaven.com/files/{asset_id}", headers=REQ_HEADERS)
            if files_response.status_code != 200:
                return {"error": f"Failed to get asset files: {files_response.status_code}"}

            files_data = files_response.json()

            # Handle different asset types
            if asset_type == "hdris":
                # For HDRIs, download the .hdr or .exr file
                if not file_format:
                    file_format = "hdr"  # Default format for HDRIs

                if "hdri" in files_data and resolution in files_data["hdri"] and file_format in files_data["hdri"][resolution]:
                    file_info = files_data["hdri"][resolution][file_format]
                    file_url = file_info["url"]

                    # For HDRIs, we need to save to a temporary file first
                    # since Blender can't properly load HDR data directly from memory
                    with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                        # Download the file
                        response = requests.get(file_url, headers=REQ_HEADERS)
                        if response.status_code != 200:
                            return {"error": f"Failed to download HDRI: {response.status_code}"}

                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name

                    try:
                        # Create a new world if none exists
                        if not bpy.data.worlds:
                            bpy.data.worlds.new("World")

                        world = bpy.data.worlds[0]
                        world.use_nodes = True
                        node_tree = world.node_tree

                        # Clear existing nodes
                        for node in node_tree.nodes:
                            node_tree.nodes.remove(node)

                        # Create nodes
                        tex_coord = node_tree.nodes.new(type='ShaderNodeTexCoord')
                        tex_coord.location = (-800, 0)

                        mapping = node_tree.nodes.new(type='ShaderNodeMapping')
                        mapping.location = (-600, 0)

                        # Load the image from the temporary file
                        env_tex = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
                        env_tex.location = (-400, 0)
                        env_tex.image = bpy.data.images.load(tmp_path)

                        # Use a color space that exists in all Blender versions
                        if file_format.lower() == 'exr':
                            # Try to use Linear color space for EXR files
                            try:
                                env_tex.image.colorspace_settings.name = 'Linear'
                            except:
                                # Fallback to Non-Color if Linear isn't available
                                env_tex.image.colorspace_settings.name = 'Non-Color'
                        else:  # hdr
                            # For HDR files, try these options in order
                            for color_space in ['Linear', 'Linear Rec.709', 'Non-Color']:
                                try:
                                    env_tex.image.colorspace_settings.name = color_space
                                    break  # Stop if we successfully set a color space
                                except:
                                    continue

                        background = node_tree.nodes.new(type='ShaderNodeBackground')
                        background.location = (-200, 0)

                        output = node_tree.nodes.new(type='ShaderNodeOutputWorld')
                        output.location = (0, 0)

                        # Connect nodes
                        node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
                        node_tree.links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
                        node_tree.links.new(env_tex.outputs['Color'], background.inputs['Color'])
                        node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

                        # Set as active world
                        bpy.context.scene.world = world

                        # Clean up temporary file
                        try:
                            tempfile._cleanup()  # This will clean up all temporary files
                        except:
                            pass

                        return {
                            "success": True,
                            "message": f"HDRI {asset_id} imported successfully",
                            "image_name": env_tex.image.name
                        }
                    except Exception as e:
                        return {"error": f"Failed to set up HDRI in Blender: {str(e)}"}
                else:
                    return {"error": f"Requested resolution or format not available for this HDRI"}

            elif asset_type == "textures":
                if not file_format:
                    file_format = "jpg"  # Default format for textures

                downloaded_maps = {}

                try:
                    for map_type in files_data:
                        if map_type not in ["blend", "gltf"]:  # Skip non-texture files
                            if resolution in files_data[map_type] and file_format in files_data[map_type][resolution]:
                                file_info = files_data[map_type][resolution][file_format]
                                file_url = file_info["url"]

                                # Use NamedTemporaryFile like we do for HDRIs
                                with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                                    # Download the file
                                    response = requests.get(file_url, headers=REQ_HEADERS)
                                    if response.status_code == 200:
                                        tmp_file.write(response.content)
                                        tmp_path = tmp_file.name

                                        # Load image from temporary file
                                        image = bpy.data.images.load(tmp_path)
                                        image.name = f"{asset_id}_{map_type}.{file_format}"

                                        # Pack the image into .blend file
                                        image.pack()

                                        # Set color space based on map type
                                        if map_type in ['color', 'diffuse', 'albedo']:
                                            try:
                                                image.colorspace_settings.name = 'sRGB'
                                            except:
                                                pass
                                        else:
                                            try:
                                                image.colorspace_settings.name = 'Non-Color'
                                            except:
                                                pass

                                        downloaded_maps[map_type] = image

                                        # Clean up temporary file
                                        try:
                                            os.unlink(tmp_path)
                                        except:
                                            pass

                    if not downloaded_maps:
                        return {"error": f"No texture maps found for the requested resolution and format"}

                    # Create a new material with the downloaded textures
                    mat = bpy.data.materials.new(name=asset_id)
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links

                    # Clear default nodes
                    for node in nodes:
                        nodes.remove(node)

                    # Create output node
                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (300, 0)

                    # Create principled BSDF node
                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (0, 0)
                    links.new(principled.outputs[0], output.inputs[0])

                    # Add texture nodes based on available maps
                    tex_coord = nodes.new(type='ShaderNodeTexCoord')
                    tex_coord.location = (-800, 0)

                    mapping = nodes.new(type='ShaderNodeMapping')
                    mapping.location = (-600, 0)
                    mapping.vector_type = 'TEXTURE'  # Changed from default 'POINT' to 'TEXTURE'
                    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

                    # Position offset for texture nodes
                    x_pos = -400
                    y_pos = 300

                    # Connect different texture maps
                    for map_type, image in downloaded_maps.items():
                        tex_node = nodes.new(type='ShaderNodeTexImage')
                        tex_node.location = (x_pos, y_pos)
                        tex_node.image = image

                        # Set color space based on map type
                        if map_type.lower() in ['color', 'diffuse', 'albedo']:
                            try:
                                tex_node.image.colorspace_settings.name = 'sRGB'
                            except:
                                pass  # Use default if sRGB not available
                        else:
                            try:
                                tex_node.image.colorspace_settings.name = 'Non-Color'
                            except:
                                pass  # Use default if Non-Color not available

                        links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])

                        # Connect to appropriate input on Principled BSDF
                        if map_type.lower() in ['color', 'diffuse', 'albedo']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                        elif map_type.lower() in ['roughness', 'rough']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
                        elif map_type.lower() in ['metallic', 'metalness', 'metal']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Metallic'])
                        elif map_type.lower() in ['normal', 'nor']:
                            # Add normal map node
                            normal_map = nodes.new(type='ShaderNodeNormalMap')
                            normal_map.location = (x_pos + 200, y_pos)
                            links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
                            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                        elif map_type in ['displacement', 'disp', 'height']:
                            # Add displacement node
                            disp_node = nodes.new(type='ShaderNodeDisplacement')
                            disp_node.location = (x_pos + 200, y_pos - 200)
                            links.new(tex_node.outputs['Color'], disp_node.inputs['Height'])
                            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])

                        y_pos -= 250

                    return {
                        "success": True,
                        "message": f"Texture {asset_id} imported as material",
                        "material": mat.name,
                        "maps": list(downloaded_maps.keys())
                    }

                except Exception as e:
                    return {"error": f"Failed to process textures: {str(e)}"}

            elif asset_type == "models":
                # For models, prefer glTF format if available
                if not file_format:
                    file_format = "gltf"  # Default format for models

                if file_format in files_data and resolution in files_data[file_format]:
                    file_info = files_data[file_format][resolution][file_format]
                    file_url = file_info["url"]

                    # Create a temporary directory to store the model and its dependencies
                    temp_dir = tempfile.mkdtemp()
                    main_file_path = ""

                    try:
                        # Download the main model file
                        main_file_name = file_url.split("/")[-1]
                        main_file_path = os.path.join(temp_dir, main_file_name)

                        response = requests.get(file_url, headers=REQ_HEADERS)
                        if response.status_code != 200:
                            return {"error": f"Failed to download model: {response.status_code}"}

                        with open(main_file_path, "wb") as f:
                            f.write(response.content)

                        # Check for included files and download them
                        if "include" in file_info and file_info["include"]:
                            for include_path, include_info in file_info["include"].items():
                                # Get the URL for the included file - this is the fix
                                include_url = include_info["url"]

                                # Create the directory structure for the included file
                                include_file_path = os.path.join(temp_dir, include_path)
                                os.makedirs(os.path.dirname(include_file_path), exist_ok=True)

                                # Download the included file
                                include_response = requests.get(include_url, headers=REQ_HEADERS)
                                if include_response.status_code == 200:
                                    with open(include_file_path, "wb") as f:
                                        f.write(include_response.content)
                                else:
                                    print(f"Failed to download included file: {include_path}")

                        # Import the model into Blender
                        if file_format == "gltf" or file_format == "glb":
                            bpy.ops.import_scene.gltf(filepath=main_file_path)
                        elif file_format == "fbx":
                            bpy.ops.import_scene.fbx(filepath=main_file_path)
                        elif file_format == "obj":
                            bpy.ops.import_scene.obj(filepath=main_file_path)
                        elif file_format == "blend":
                            # For blend files, we need to append or link
                            with bpy.data.libraries.load(main_file_path, link=False) as (data_from, data_to):
                                data_to.objects = data_from.objects

                            # Link the objects to the scene
                            for obj in data_to.objects:
                                if obj is not None:
                                    bpy.context.collection.objects.link(obj)
                        else:
                            return {"error": f"Unsupported model format: {file_format}"}

                        # Get the names of imported objects
                        imported_objects = [obj.name for obj in bpy.context.selected_objects]

                        return {
                            "success": True,
                            "message": f"Model {asset_id} imported successfully",
                            "imported_objects": imported_objects
                        }
                    except Exception as e:
                        return {"error": f"Failed to import model: {str(e)}"}
                    finally:
                        # Clean up temporary directory
                        with suppress(Exception):
                            shutil.rmtree(temp_dir)
                else:
                    return {"error": f"Requested format or resolution not available for this model"}

            else:
                return {"error": f"Unsupported asset type: {asset_type}"}

        except Exception as e:
            return {"error": f"Failed to download asset: {str(e)}"}

    def set_texture(self, object_name, texture_id):
        """Apply a previously downloaded Polyhaven texture to an object by creating a new material"""
        try:
            # Get the object
            obj = bpy.data.objects.get(object_name)
            if not obj:
                return {"error": f"Object not found: {object_name}"}

            # Make sure object can accept materials
            if not hasattr(obj, 'data') or not hasattr(obj.data, 'materials'):
                return {"error": f"Object {object_name} cannot accept materials"}

            # Find all images related to this texture and ensure they're properly loaded
            texture_images = {}
            for img in bpy.data.images:
                if img.name.startswith(texture_id + "_"):
                    # Extract the map type from the image name
                    map_type = img.name.split('_')[-1].split('.')[0]

                    # Force a reload of the image
                    img.reload()

                    # Ensure proper color space
                    if map_type.lower() in ['color', 'diffuse', 'albedo']:
                        try:
                            img.colorspace_settings.name = 'sRGB'
                        except:
                            pass
                    else:
                        try:
                            img.colorspace_settings.name = 'Non-Color'
                        except:
                            pass

                    # Ensure the image is packed
                    if not img.packed_file:
                        img.pack()

                    texture_images[map_type] = img
                    print(f"Loaded texture map: {map_type} - {img.name}")

                    # Debug info
                    print(f"Image size: {img.size[0]}x{img.size[1]}")
                    print(f"Color space: {img.colorspace_settings.name}")
                    print(f"File format: {img.file_format}")
                    print(f"Is packed: {bool(img.packed_file)}")

            if not texture_images:
                return {"error": f"No texture images found for: {texture_id}. Please download the texture first."}

            # Create a new material
            new_mat_name = f"{texture_id}_material_{object_name}"

            # Remove any existing material with this name to avoid conflicts
            existing_mat = bpy.data.materials.get(new_mat_name)
            if existing_mat:
                bpy.data.materials.remove(existing_mat)

            new_mat = bpy.data.materials.new(name=new_mat_name)
            new_mat.use_nodes = True

            # Set up the material nodes
            nodes = new_mat.node_tree.nodes
            links = new_mat.node_tree.links

            # Clear default nodes
            nodes.clear()

            # Create output node
            output = nodes.new(type='ShaderNodeOutputMaterial')
            output.location = (600, 0)

            # Create principled BSDF node
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            principled.location = (300, 0)
            links.new(principled.outputs[0], output.inputs[0])

            # Add texture nodes based on available maps
            tex_coord = nodes.new(type='ShaderNodeTexCoord')
            tex_coord.location = (-800, 0)

            mapping = nodes.new(type='ShaderNodeMapping')
            mapping.location = (-600, 0)
            mapping.vector_type = 'TEXTURE'  # Changed from default 'POINT' to 'TEXTURE'
            links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

            # Position offset for texture nodes
            x_pos = -400
            y_pos = 300

            # Connect different texture maps
            for map_type, image in texture_images.items():
                tex_node = nodes.new(type='ShaderNodeTexImage')
                tex_node.location = (x_pos, y_pos)
                tex_node.image = image

                # Set color space based on map type
                if map_type.lower() in ['color', 'diffuse', 'albedo']:
                    try:
                        tex_node.image.colorspace_settings.name = 'sRGB'
                    except:
                        pass  # Use default if sRGB not available
                else:
                    try:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    except:
                        pass  # Use default if Non-Color not available

                links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])

                # Connect to appropriate input on Principled BSDF
                if map_type.lower() in ['color', 'diffuse', 'albedo']:
                    links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                elif map_type.lower() in ['roughness', 'rough']:
                    links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
                elif map_type.lower() in ['metallic', 'metalness', 'metal']:
                    links.new(tex_node.outputs['Color'], principled.inputs['Metallic'])
                elif map_type.lower() in ['normal', 'nor', 'dx', 'gl']:
                    # Add normal map node
                    normal_map = nodes.new(type='ShaderNodeNormalMap')
                    normal_map.location = (x_pos + 200, y_pos)
                    links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
                    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                elif map_type.lower() in ['displacement', 'disp', 'height']:
                    # Add displacement node
                    disp_node = nodes.new(type='ShaderNodeDisplacement')
                    disp_node.location = (x_pos + 200, y_pos - 200)
                    disp_node.inputs['Scale'].default_value = 0.1  # Reduce displacement strength
                    links.new(tex_node.outputs['Color'], disp_node.inputs['Height'])
                    links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])

                y_pos -= 250

            # Second pass: Connect nodes with proper handling for special cases
            texture_nodes = {}

            # First find all texture nodes and store them by map type
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    for map_type, image in texture_images.items():
                        if node.image == image:
                            texture_nodes[map_type] = node
                            break

            # Now connect everything using the nodes instead of images
            # Handle base color (diffuse)
            for map_name in ['color', 'diffuse', 'albedo']:
                if map_name in texture_nodes:
                    links.new(texture_nodes[map_name].outputs['Color'], principled.inputs['Base Color'])
                    print(f"Connected {map_name} to Base Color")
                    break

            # Handle roughness
            for map_name in ['roughness', 'rough']:
                if map_name in texture_nodes:
                    links.new(texture_nodes[map_name].outputs['Color'], principled.inputs['Roughness'])
                    print(f"Connected {map_name} to Roughness")
                    break

            # Handle metallic
            for map_name in ['metallic', 'metalness', 'metal']:
                if map_name in texture_nodes:
                    links.new(texture_nodes[map_name].outputs['Color'], principled.inputs['Metallic'])
                    print(f"Connected {map_name} to Metallic")
                    break

            # Handle normal maps
            for map_name in ['gl', 'dx', 'nor']:
                if map_name in texture_nodes:
                    normal_map_node = nodes.new(type='ShaderNodeNormalMap')
                    normal_map_node.location = (100, 100)
                    links.new(texture_nodes[map_name].outputs['Color'], normal_map_node.inputs['Color'])
                    links.new(normal_map_node.outputs['Normal'], principled.inputs['Normal'])
                    print(f"Connected {map_name} to Normal")
                    break

            # Handle displacement
            for map_name in ['displacement', 'disp', 'height']:
                if map_name in texture_nodes:
                    disp_node = nodes.new(type='ShaderNodeDisplacement')
                    disp_node.location = (300, -200)
                    disp_node.inputs['Scale'].default_value = 0.1  # Reduce displacement strength
                    links.new(texture_nodes[map_name].outputs['Color'], disp_node.inputs['Height'])
                    links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
                    print(f"Connected {map_name} to Displacement")
                    break

            # Handle ARM texture (Ambient Occlusion, Roughness, Metallic)
            if 'arm' in texture_nodes:
                separate_rgb = nodes.new(type='ShaderNodeSeparateRGB')
                separate_rgb.location = (-200, -100)
                links.new(texture_nodes['arm'].outputs['Color'], separate_rgb.inputs['Image'])

                # Connect Roughness (G) if no dedicated roughness map
                if not any(map_name in texture_nodes for map_name in ['roughness', 'rough']):
                    links.new(separate_rgb.outputs['G'], principled.inputs['Roughness'])
                    print("Connected ARM.G to Roughness")

                # Connect Metallic (B) if no dedicated metallic map
                if not any(map_name in texture_nodes for map_name in ['metallic', 'metalness', 'metal']):
                    links.new(separate_rgb.outputs['B'], principled.inputs['Metallic'])
                    print("Connected ARM.B to Metallic")

                # For AO (R channel), multiply with base color if we have one
                base_color_node = None
                for map_name in ['color', 'diffuse', 'albedo']:
                    if map_name in texture_nodes:
                        base_color_node = texture_nodes[map_name]
                        break

                if base_color_node:
                    mix_node = nodes.new(type='ShaderNodeMixRGB')
                    mix_node.location = (100, 200)
                    mix_node.blend_type = 'MULTIPLY'
                    mix_node.inputs['Fac'].default_value = 0.8  # 80% influence

                    # Disconnect direct connection to base color
                    for link in base_color_node.outputs['Color'].links:
                        if link.to_socket == principled.inputs['Base Color']:
                            links.remove(link)

                    # Connect through the mix node
                    links.new(base_color_node.outputs['Color'], mix_node.inputs[1])
                    links.new(separate_rgb.outputs['R'], mix_node.inputs[2])
                    links.new(mix_node.outputs['Color'], principled.inputs['Base Color'])
                    print("Connected ARM.R to AO mix with Base Color")

            # Handle AO (Ambient Occlusion) if separate
            if 'ao' in texture_nodes:
                base_color_node = None
                for map_name in ['color', 'diffuse', 'albedo']:
                    if map_name in texture_nodes:
                        base_color_node = texture_nodes[map_name]
                        break

                if base_color_node:
                    mix_node = nodes.new(type='ShaderNodeMixRGB')
                    mix_node.location = (100, 200)
                    mix_node.blend_type = 'MULTIPLY'
                    mix_node.inputs['Fac'].default_value = 0.8  # 80% influence

                    # Disconnect direct connection to base color
                    for link in base_color_node.outputs['Color'].links:
                        if link.to_socket == principled.inputs['Base Color']:
                            links.remove(link)

                    # Connect through the mix node
                    links.new(base_color_node.outputs['Color'], mix_node.inputs[1])
                    links.new(texture_nodes['ao'].outputs['Color'], mix_node.inputs[2])
                    links.new(mix_node.outputs['Color'], principled.inputs['Base Color'])
                    print("Connected AO to mix with Base Color")

            # CRITICAL: Make sure to clear all existing materials from the object
            while len(obj.data.materials) > 0:
                obj.data.materials.pop(index=0)

            # Assign the new material to the object
            obj.data.materials.append(new_mat)

            # CRITICAL: Make the object active and select it
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

            # CRITICAL: Force Blender to update the material
            bpy.context.view_layer.update()

            # Get the list of texture maps
            texture_maps = list(texture_images.keys())

            # Get info about texture nodes for debugging
            material_info = {
                "name": new_mat.name,
                "has_nodes": new_mat.use_nodes,
                "node_count": len(new_mat.node_tree.nodes),
                "texture_nodes": []
            }

            for node in new_mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    connections = []
                    for output in node.outputs:
                        for link in output.links:
                            connections.append(f"{output.name} → {link.to_node.name}.{link.to_socket.name}")

                    material_info["texture_nodes"].append({
                        "name": node.name,
                        "image": node.image.name,
                        "colorspace": node.image.colorspace_settings.name,
                        "connections": connections
                    })

            return {
                "success": True,
                "message": f"Created new material and applied texture {texture_id} to {object_name}",
                "material": new_mat.name,
                "maps": texture_maps,
                "material_info": material_info
            }

        except Exception as e:
            print(f"Error in set_texture: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to apply texture: {str(e)}"}



    def get_polyhaven_status(self):
        """Get the current status of PolyHaven integration"""
        enabled = bpy.context.scene.blendermcp_use_polyhaven
        if enabled:
            return {"enabled": True, "message": "PolyHaven integration is enabled and ready to use."}
        else:
            return {
                "enabled": False,
                "message": """PolyHaven integration is currently disabled. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Check the 'Use assets from Poly Haven' checkbox
                            3. Restart the connection to Claude"""
        }

    #region Hyper3D
    def get_hyper3d_status(self):
        """Get the current status of Hyper3D Rodin integration"""
        enabled = bpy.context.scene.blendermcp_use_hyper3d
        if enabled:
            if not bpy.context.scene.blendermcp_hyper3d_api_key:
                return {
                    "enabled": False,
                    "message": """Hyper3D Rodin integration is currently enabled, but API key is not given. To enable it:
                                1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                                2. Keep the 'Use Hyper3D Rodin 3D model generation' checkbox checked
                                3. Choose the right plaform and fill in the API Key
                                4. Restart the connection to Claude"""
                }
            mode = bpy.context.scene.blendermcp_hyper3d_mode
            message = f"Hyper3D Rodin integration is enabled and ready to use. Mode: {mode}. " + \
                f"Key type: {'private' if bpy.context.scene.blendermcp_hyper3d_api_key != RODIN_FREE_TRIAL_KEY else 'free_trial'}"
            return {
                "enabled": True,
                "message": message
            }
        else:
            return {
                "enabled": False,
                "message": """Hyper3D Rodin integration is currently disabled. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Check the 'Use Hyper3D Rodin 3D model generation' checkbox
                            3. Restart the connection to Claude"""
            }

    def create_rodin_job(self, *args, **kwargs):
        match bpy.context.scene.blendermcp_hyper3d_mode:
            case "MAIN_SITE":
                return self.create_rodin_job_main_site(*args, **kwargs)
            case "FAL_AI":
                return self.create_rodin_job_fal_ai(*args, **kwargs)
            case _:
                return f"Error: Unknown Hyper3D Rodin mode!"

    def create_rodin_job_main_site(
            self,
            text_prompt: str=None,
            images: list[tuple[str, str]]=None,
            bbox_condition=None
        ):
        try:
            if images is None:
                images = []
            """Call Rodin API, get the job uuid and subscription key"""
            files = [
                *[("images", (f"{i:04d}{img_suffix}", img)) for i, (img_suffix, img) in enumerate(images)],
                ("tier", (None, "Sketch")),
                ("mesh_mode", (None, "Raw")),
            ]
            if text_prompt:
                files.append(("prompt", (None, text_prompt)))
            if bbox_condition:
                files.append(("bbox_condition", (None, json.dumps(bbox_condition))))
            response = requests.post(
                "https://hyperhuman.deemos.com/api/v2/rodin",
                headers={
                    "Authorization": f"Bearer {bpy.context.scene.blendermcp_hyper3d_api_key}",
                },
                files=files
            )
            data = response.json()
            return data
        except Exception as e:
            return {"error": str(e)}

    def create_rodin_job_fal_ai(
            self,
            text_prompt: str=None,
            images: list[tuple[str, str]]=None,
            bbox_condition=None
        ):
        try:
            req_data = {
                "tier": "Sketch",
            }
            if images:
                req_data["input_image_urls"] = images
            if text_prompt:
                req_data["prompt"] = text_prompt
            if bbox_condition:
                req_data["bbox_condition"] = bbox_condition
            response = requests.post(
                "https://queue.fal.run/fal-ai/hyper3d/rodin",
                headers={
                    "Authorization": f"Key {bpy.context.scene.blendermcp_hyper3d_api_key}",
                    "Content-Type": "application/json",
                },
                json=req_data
            )
            data = response.json()
            return data
        except Exception as e:
            return {"error": str(e)}

    def poll_rodin_job_status(self, *args, **kwargs):
        match bpy.context.scene.blendermcp_hyper3d_mode:
            case "MAIN_SITE":
                return self.poll_rodin_job_status_main_site(*args, **kwargs)
            case "FAL_AI":
                return self.poll_rodin_job_status_fal_ai(*args, **kwargs)
            case _:
                return f"Error: Unknown Hyper3D Rodin mode!"

    def poll_rodin_job_status_main_site(self, subscription_key: str):
        """Call the job status API to get the job status"""
        response = requests.post(
            "https://hyperhuman.deemos.com/api/v2/status",
            headers={
                "Authorization": f"Bearer {bpy.context.scene.blendermcp_hyper3d_api_key}",
            },
            json={
                "subscription_key": subscription_key,
            },
        )
        data = response.json()
        return {
            "status_list": [i["status"] for i in data["jobs"]]
        }

    def poll_rodin_job_status_fal_ai(self, request_id: str):
        """Call the job status API to get the job status"""
        response = requests.get(
            f"https://queue.fal.run/fal-ai/hyper3d/requests/{request_id}/status",
            headers={
                "Authorization": f"KEY {bpy.context.scene.blendermcp_hyper3d_api_key}",
            },
        )
        data = response.json()
        return data

    @staticmethod
    def _clean_imported_glb(filepath, mesh_name=None):
        # Get the set of existing objects before import
        existing_objects = set(bpy.data.objects)

        # Import the GLB file
        bpy.ops.import_scene.gltf(filepath=filepath)

        # Ensure the context is updated
        bpy.context.view_layer.update()

        # Get all imported objects
        imported_objects = list(set(bpy.data.objects) - existing_objects)
        # imported_objects = [obj for obj in bpy.context.view_layer.objects if obj.select_get()]

        if not imported_objects:
            print("Error: No objects were imported.")
            return

        # Identify the mesh object
        mesh_obj = None

        if len(imported_objects) == 1 and imported_objects[0].type == 'MESH':
            mesh_obj = imported_objects[0]
            print("Single mesh imported, no cleanup needed.")
        else:
            if len(imported_objects) == 2:
                empty_objs = [i for i in imported_objects if i.type == "EMPTY"]
                if len(empty_objs) != 1:
                    print("Error: Expected an empty node with one mesh child or a single mesh object.")
                    return
                parent_obj = empty_objs.pop()
                if len(parent_obj.children) == 1:
                    potential_mesh = parent_obj.children[0]
                    if potential_mesh.type == 'MESH':
                        print("GLB structure confirmed: Empty node with one mesh child.")

                        # Unparent the mesh from the empty node
                        potential_mesh.parent = None

                        # Remove the empty node
                        bpy.data.objects.remove(parent_obj)
                        print("Removed empty node, keeping only the mesh.")

                        mesh_obj = potential_mesh
                    else:
                        print("Error: Child is not a mesh object.")
                        return
                else:
                    print("Error: Expected an empty node with one mesh child or a single mesh object.")
                    return
            else:
                print("Error: Expected an empty node with one mesh child or a single mesh object.")
                return

        # Rename the mesh if needed
        try:
            if mesh_obj and mesh_obj.name is not None and mesh_name:
                mesh_obj.name = mesh_name
                if mesh_obj.data.name is not None:
                    mesh_obj.data.name = mesh_name
                print(f"Mesh renamed to: {mesh_name}")
        except Exception as e:
            print("Having issue with renaming, give up renaming.")

        return mesh_obj

    def import_generated_asset(self, *args, **kwargs):
        match bpy.context.scene.blendermcp_hyper3d_mode:
            case "MAIN_SITE":
                return self.import_generated_asset_main_site(*args, **kwargs)
            case "FAL_AI":
                return self.import_generated_asset_fal_ai(*args, **kwargs)
            case _:
                return f"Error: Unknown Hyper3D Rodin mode!"

    def import_generated_asset_main_site(self, task_uuid: str, name: str):
        """Fetch the generated asset, import into blender"""
        response = requests.post(
            "https://hyperhuman.deemos.com/api/v2/download",
            headers={
                "Authorization": f"Bearer {bpy.context.scene.blendermcp_hyper3d_api_key}",
            },
            json={
                'task_uuid': task_uuid
            }
        )
        data_ = response.json()
        temp_file = None
        for i in data_["list"]:
            if i["name"].endswith(".glb"):
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix=task_uuid,
                    suffix=".glb",
                )

                try:
                    # Download the content
                    response = requests.get(i["url"], stream=True)
                    response.raise_for_status()  # Raise an exception for HTTP errors

                    # Write the content to the temporary file
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)

                    # Close the file
                    temp_file.close()

                except Exception as e:
                    # Clean up the file if there's an error
                    temp_file.close()
                    os.unlink(temp_file.name)
                    return {"succeed": False, "error": str(e)}

                break
        else:
            return {"succeed": False, "error": "Generation failed. Please first make sure that all jobs of the task are done and then try again later."}

        try:
            obj = self._clean_imported_glb(
                filepath=temp_file.name,
                mesh_name=name
            )
            result = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            }

            if obj.type == "MESH":
                bounding_box = self._get_aabb(obj)
                result["world_bounding_box"] = bounding_box

            return {
                "succeed": True, **result
            }
        except Exception as e:
            return {"succeed": False, "error": str(e)}

    def import_generated_asset_fal_ai(self, request_id: str, name: str):
        """Fetch the generated asset, import into blender"""
        response = requests.get(
            f"https://queue.fal.run/fal-ai/hyper3d/requests/{request_id}",
            headers={
                "Authorization": f"Key {bpy.context.scene.blendermcp_hyper3d_api_key}",
            }
        )
        data_ = response.json()
        temp_file = None

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            prefix=request_id,
            suffix=".glb",
        )

        try:
            # Download the content
            response = requests.get(data_["model_mesh"]["url"], stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Write the content to the temporary file
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

            # Close the file
            temp_file.close()

        except Exception as e:
            # Clean up the file if there's an error
            temp_file.close()
            os.unlink(temp_file.name)
            return {"succeed": False, "error": str(e)}

        try:
            obj = self._clean_imported_glb(
                filepath=temp_file.name,
                mesh_name=name
            )
            result = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            }

            if obj.type == "MESH":
                bounding_box = self._get_aabb(obj)
                result["world_bounding_box"] = bounding_box

            return {
                "succeed": True, **result
            }
        except Exception as e:
            return {"succeed": False, "error": str(e)}
    #endregion
 
    #region Sketchfab API
    def get_sketchfab_status(self):
        """Get the current status of Sketchfab integration"""
        enabled = bpy.context.scene.blendermcp_use_sketchfab
        api_key = bpy.context.scene.blendermcp_sketchfab_api_key

        # Test the API key if present
        if api_key:
            try:
                headers = {
                    "Authorization": f"Token {api_key}"
                }

                response = requests.get(
                    "https://api.sketchfab.com/v3/me",
                    headers=headers,
                    timeout=30  # Add timeout of 30 seconds
                )

                if response.status_code == 200:
                    user_data = response.json()
                    username = user_data.get("username", "Unknown user")
                    return {
                        "enabled": True,
                        "message": f"Sketchfab integration is enabled and ready to use. Logged in as: {username}"
                    }
                else:
                    return {
                        "enabled": False,
                        "message": f"Sketchfab API key seems invalid. Status code: {response.status_code}"
                    }
            except requests.exceptions.Timeout:
                return {
                    "enabled": False,
                    "message": "Timeout connecting to Sketchfab API. Check your internet connection."
                }
            except Exception as e:
                return {
                    "enabled": False,
                    "message": f"Error testing Sketchfab API key: {str(e)}"
                }

        if enabled and api_key:
            return {"enabled": True, "message": "Sketchfab integration is enabled and ready to use."}
        elif enabled and not api_key:
            return {
                "enabled": False,
                "message": """Sketchfab integration is currently enabled, but API key is not given. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Keep the 'Use Sketchfab' checkbox checked
                            3. Enter your Sketchfab API Key
                            4. Restart the connection to Claude"""
            }
        else:
            return {
                "enabled": False,
                "message": """Sketchfab integration is currently disabled. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Check the 'Use assets from Sketchfab' checkbox
                            3. Enter your Sketchfab API Key
                            4. Restart the connection to Claude"""
            }

    def get_polypizza_status(self):
        """Get the current status of Poly Pizza integration."""
        enabled = bpy.context.scene.blendermcp_use_polypizza
        api_key = bpy.context.scene.blendermcp_polypizza_api_key

        if enabled and api_key:
            return {
                "enabled": True,
                "message": "Poly Pizza integration is enabled and ready to use."
            }
        elif enabled and not api_key:
            return {
                "enabled": False,
                "message": """Poly Pizza integration is enabled, but API key is missing. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Keep the 'Use assets from Poly Pizza' checkbox checked
                            3. Enter your Poly Pizza API Key
                            4. Restart the connection to Claude"""
            }
        else:
            return {
                "enabled": False,
                "message": """Poly Pizza integration is currently disabled. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Check the 'Use assets from Poly Pizza' checkbox
                            3. Enter your Poly Pizza API Key
                            4. Restart the connection to Claude"""
            }

    def _extract_polypizza_results(self, payload):
        """Normalize Poly Pizza API response into a stable list."""
        if not isinstance(payload, dict):
            return []

        candidates = []
        for key in ("results", "models", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break

        normalized = []
        for item in candidates:
            if not isinstance(item, dict):
                continue

            model_id = item.get("id") or item.get("uid") or item.get("slug")
            title = item.get("title") or item.get("name") or model_id or "Unknown"
            tri_count = item.get("triCount") or item.get("tri_count") or item.get("tris")
            license_label = item.get("license") or item.get("licenseLabel") or "Unknown"

            download_url = (
                item.get("download")
                or item.get("downloadUrl")
                or item.get("glb")
                or item.get("modelUrl")
            )

            thumb = item.get("thumbnail") or item.get("thumbnailUrl") or item.get("preview")

            normalized.append({
                "id": model_id,
                "name": title,
                "triCount": tri_count,
                "license": license_label,
                "download": download_url,
                "thumbnail": thumb,
            })

        return normalized

    def _search_polypizza_html_fallback(self, query, count=20):
        """Fallback search by parsing public Poly Pizza pages when API endpoints fail.

        Quality goal:
        - Return query-relevant assets only (avoid unrelated suggestions)
        - Enrich with model-page metadata (name, thumbnail, download URL when available)
        """
        try:
            url = f"https://poly.pizza/search/{requests.utils.quote(query)}"
            res = requests.get(url, headers={"User-Agent": "blender-mcp"}, timeout=30)
            if res.status_code != 200:
                return []

            page_html = res.text

            # Query tokens for strict relevance filtering
            tokens = [t.lower() for t in re.findall(r'[A-Za-z0-9]+', query or "") if len(t) >= 2]

            # Collect unique ids from /m/<id> links (cap candidates for speed)
            candidate_ids = []
            for mid in re.findall(r'/m/([A-Za-z0-9_-]+)', page_html):
                if mid not in candidate_ids:
                    candidate_ids.append(mid)
                if len(candidate_ids) >= max(40, count * 4):
                    break

            scored = []
            for mid in candidate_ids:
                meta = self._get_polypizza_model_meta(mid)
                if not meta:
                    continue

                name = (meta.get("name") or mid)
                search_text = f"{name} {mid}".lower()

                # Strict relevance: only keep items that match query tokens
                score = 0
                if tokens:
                    for tok in tokens:
                        if tok in search_text:
                            score += 1
                    if score == 0:
                        continue
                else:
                    score = 1

                scored.append((score, {
                    "id": mid,
                    "name": name,
                    "triCount": None,
                    "license": meta.get("license", "Unknown"),
                    "download": meta.get("download"),
                    "thumbnail": meta.get("thumbnail"),
                }))

            # Best matches first
            scored.sort(key=lambda x: (-x[0], x[1].get("name", "")))
            return [item for _, item in scored[:count]]
        except Exception:
            return []

    def _get_polypizza_model_meta(self, model_id):
        """Get model metadata from model page for fallback search quality."""
        try:
            page_url = f"https://poly.pizza/m/{model_id}"
            res = requests.get(page_url, headers={"User-Agent": "blender-mcp"}, timeout=30)
            if res.status_code != 200:
                return None

            page_html = html.unescape(res.text)

            # Try OpenGraph title first
            name = None
            m_title = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']', page_html, flags=re.IGNORECASE)
            if m_title:
                name = m_title.group(1).strip()

            # Optional page title fallback
            if not name:
                t = re.search(r'<title>([^<]+)</title>', page_html, flags=re.IGNORECASE)
                if t:
                    name = t.group(1).strip()

            download = self._resolve_polypizza_download_from_model_page(model_id)

            # Thumbnail from OpenGraph image if present
            thumbnail = None
            m_img = re.search(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', page_html, flags=re.IGNORECASE)
            if m_img:
                thumbnail = m_img.group(1).strip()

            # Very light license signal if present in page
            lic = "Unknown"
            if re.search(r'CC0', page_html, flags=re.IGNORECASE):
                lic = "CC0"
            elif re.search(r'CC-BY|Attribution', page_html, flags=re.IGNORECASE):
                lic = "CC-BY"

            return {
                "id": model_id,
                "name": name or model_id,
                "download": download,
                "thumbnail": thumbnail,
                "license": lic,
            }
        except Exception:
            return None

    def _resolve_polypizza_download_from_model_page(self, model_id):
        """Resolve direct GLB URL from model page as a fallback."""
        try:
            page_url = f"https://poly.pizza/m/{model_id}"
            res = requests.get(page_url, headers={"User-Agent": "blender-mcp"}, timeout=30)
            if res.status_code != 200:
                return None

            page_html = html.unescape(res.text)

            # Prefer static Poly Pizza binary links directly.
            static_matches = re.findall(
                r'https://static\.poly\.pizza/[^\"\'\s>]+\.glb(?:\.br)?',
                page_html,
                flags=re.IGNORECASE,
            )
            if static_matches:
                # Prefer pure .glb over .glb.br when both exist.
                for url in static_matches:
                    if url.lower().endswith('.glb'):
                        return url
                # Fallback: strip .br suffix if present.
                candidate = static_matches[0]
                if candidate.lower().endswith('.glb.br'):
                    return candidate[:-3]
                return candidate

            # Some pages embed viewer URLs with src=<glb_url> query param.
            src_param_match = re.search(
                r'src=(https://static\.poly\.pizza/[^&\"\'\s>]+\.glb)',
                page_html,
                flags=re.IGNORECASE,
            )
            if src_param_match:
                return src_param_match.group(1)

            # Generic absolute .glb fallback
            match = re.search(r'https://[^\"\']+\.glb', page_html, flags=re.IGNORECASE)
            if match:
                return match.group(0)

            # Try relative GLB path fallback
            rel = re.search(r'(/[^\"\']+\.glb)', page_html, flags=re.IGNORECASE)
            if rel:
                return f"https://poly.pizza{rel.group(1)}"

            return None
        except Exception:
            return None

    def search_polypizza_models(self, query, count=20):
        """Search Poly Pizza models using API key auth."""
        try:
            api_key = bpy.context.scene.blendermcp_polypizza_api_key
            if not api_key:
                return {"error": "Poly Pizza API key is not configured"}

            count = max(1, min(int(count), 50))

            headers = {
                "x-auth-token": api_key,
                "User-Agent": "blender-mcp",
                "Accept": "application/json",
            }

            endpoint_candidates = [
                (f"{POLYPIZZA_API_BASE}/search", {"query": query, "count": count}),
                (f"{POLYPIZZA_API_BASE}/search", {"q": query, "count": count}),
                (f"{POLYPIZZA_API_BASE}/models/search", {"query": query, "count": count}),
                (f"{POLYPIZZA_API_BASE}/models", {"search": query, "count": count}),
            ]

            last_error = None
            for url, params in endpoint_candidates:
                try:
                    res = requests.get(url, headers=headers, params=params, timeout=30)
                    if res.status_code != 200:
                        last_error = f"HTTP {res.status_code} on {url}"
                        continue
                    payload = res.json()
                    normalized = self._extract_polypizza_results(payload)
                    if normalized:
                        return {
                            "results": normalized[:count],
                            "count": len(normalized[:count]),
                        }
                except Exception as e:
                    last_error = str(e)
                    continue

            # Fallback: parse public search page
            fallback_results = self._search_polypizza_html_fallback(query, count=count)
            if fallback_results:
                return {
                    "results": fallback_results,
                    "count": len(fallback_results),
                    "source": "html_fallback",
                }

            return {"error": f"Poly Pizza search failed: {last_error or 'unknown error'}"}
        except Exception as e:
            return {"error": str(e)}

    def download_polypizza_model(self, model_id=None, download_url=None, target_size=1.0):
        """Download/import a Poly Pizza model from direct URL or model id lookup."""
        try:
            api_key = bpy.context.scene.blendermcp_polypizza_api_key
            if not api_key:
                return {"error": "Poly Pizza API key is not configured"}

            headers = {
                "x-auth-token": api_key,
                "User-Agent": "blender-mcp",
                "Accept": "application/json",
            }

            resolved_url = download_url
            if not resolved_url and model_id:
                detail_candidates = [
                    f"{POLYPIZZA_API_BASE}/models/{model_id}",
                    f"{POLYPIZZA_API_BASE}/model/{model_id}",
                ]
                for url in detail_candidates:
                    try:
                        res = requests.get(url, headers=headers, timeout=30)
                        if res.status_code != 200:
                            continue
                        payload = res.json()
                        normalized = self._extract_polypizza_results(payload)
                        if normalized and normalized[0].get("download"):
                            resolved_url = normalized[0].get("download")
                            break
                        if isinstance(payload, dict):
                            resolved_url = payload.get("download") or payload.get("downloadUrl") or payload.get("glb")
                            if resolved_url:
                                break
                    except Exception:
                        continue

                if not resolved_url:
                    resolved_url = self._resolve_polypizza_download_from_model_page(model_id)

            # Handle compressed variant URLs gracefully
            if resolved_url and resolved_url.lower().endswith('.glb.br'):
                resolved_url = resolved_url[:-3]

            if not resolved_url:
                return {"error": "No downloadable URL resolved for Poly Pizza model"}

            temp_dir = tempfile.mkdtemp(prefix="polypizza_")
            try:
                file_name = (model_id or "polypizza_model") + ".glb"
                file_path = os.path.join(temp_dir, file_name)

                dl = requests.get(resolved_url, timeout=60)
                if dl.status_code != 200:
                    return {"error": f"Download failed with status {dl.status_code}"}
                with open(file_path, "wb") as f:
                    f.write(dl.content)

                existing_objects = set(bpy.data.objects)
                bpy.ops.import_scene.gltf(filepath=file_path)
                imported_objects = list(set(bpy.data.objects) - existing_objects)
                imported_names = [obj.name for obj in imported_objects]

                # Optional normalization by scaling roots so max dimension == target_size
                roots = [obj for obj in imported_objects if obj.parent is None]
                all_meshes = []
                for root in roots:
                    stack = [root]
                    while stack:
                        n = stack.pop()
                        if n.type == 'MESH':
                            all_meshes.append(n)
                        stack.extend(list(n.children))

                scale_applied = 1.0
                if all_meshes and target_size and float(target_size) > 0:
                    all_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
                    all_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
                    for mesh_obj in all_meshes:
                        for corner in mesh_obj.bound_box:
                            wc = mesh_obj.matrix_world @ mathutils.Vector(corner)
                            all_min.x = min(all_min.x, wc.x)
                            all_min.y = min(all_min.y, wc.y)
                            all_min.z = min(all_min.z, wc.z)
                            all_max.x = max(all_max.x, wc.x)
                            all_max.y = max(all_max.y, wc.y)
                            all_max.z = max(all_max.z, wc.z)

                    dims = [all_max.x - all_min.x, all_max.y - all_min.y, all_max.z - all_min.z]
                    max_dim = max(dims) if dims else 0.0
                    if max_dim > 0:
                        scale_applied = float(target_size) / max_dim
                        for root in roots:
                            root.scale = (
                                root.scale.x * scale_applied,
                                root.scale.y * scale_applied,
                                root.scale.z * scale_applied,
                            )
                        bpy.context.view_layer.update()

                return {
                    "success": True,
                    "imported_objects": imported_names,
                    "scale_applied": round(scale_applied, 6),
                    "normalized": True if scale_applied != 1.0 else False,
                }
            finally:
                with suppress(Exception):
                    shutil.rmtree(temp_dir)
        except Exception as e:
            return {"error": f"Failed to download Poly Pizza model: {str(e)}"}

    def search_sketchfab_models(self, query, categories=None, count=20, downloadable=True):
        """Search for models on Sketchfab based on query and optional filters"""
        try:
            api_key = bpy.context.scene.blendermcp_sketchfab_api_key
            if not api_key:
                return {"error": "Sketchfab API key is not configured"}

            # Build search parameters with exact fields from Sketchfab API docs
            params = {
                "type": "models",
                "q": query,
                "count": count,
                "downloadable": downloadable,
                "archives_flavours": False
            }

            if categories:
                params["categories"] = categories

            # Make API request to Sketchfab search endpoint
            # The proper format according to Sketchfab API docs for API key auth
            headers = {
                "Authorization": f"Token {api_key}"
            }


            # Use the search endpoint as specified in the API documentation
            response = requests.get(
                "https://api.sketchfab.com/v3/search",
                headers=headers,
                params=params,
                timeout=30  # Add timeout of 30 seconds
            )

            if response.status_code == 401:
                return {"error": "Authentication failed (401). Check your API key."}

            if response.status_code != 200:
                return {"error": f"API request failed with status code {response.status_code}"}

            response_data = response.json()

            # Safety check on the response structure
            if response_data is None:
                return {"error": "Received empty response from Sketchfab API"}

            # Handle 'results' potentially missing from response
            results = response_data.get("results", [])
            if not isinstance(results, list):
                return {"error": f"Unexpected response format from Sketchfab API: {response_data}"}

            return response_data

        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Check your internet connection."}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response from Sketchfab API: {str(e)}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_sketchfab_model_preview(self, uid):
        """Get thumbnail preview image of a Sketchfab model by its UID"""
        try:
            import base64
            
            api_key = bpy.context.scene.blendermcp_sketchfab_api_key
            if not api_key:
                return {"error": "Sketchfab API key is not configured"}

            headers = {"Authorization": f"Token {api_key}"}
            
            # Get model info which includes thumbnails
            response = requests.get(
                f"https://api.sketchfab.com/v3/models/{uid}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 401:
                return {"error": "Authentication failed (401). Check your API key."}
            
            if response.status_code == 404:
                return {"error": f"Model not found: {uid}"}
            
            if response.status_code != 200:
                return {"error": f"Failed to get model info: {response.status_code}"}
            
            data = response.json()
            thumbnails = data.get("thumbnails", {}).get("images", [])
            
            if not thumbnails:
                return {"error": "No thumbnail available for this model"}
            
            # Find a suitable thumbnail (prefer medium size ~640px)
            selected_thumbnail = None
            for thumb in thumbnails:
                width = thumb.get("width", 0)
                if 400 <= width <= 800:
                    selected_thumbnail = thumb
                    break
            
            # Fallback to the first available thumbnail
            if not selected_thumbnail:
                selected_thumbnail = thumbnails[0]
            
            thumbnail_url = selected_thumbnail.get("url")
            if not thumbnail_url:
                return {"error": "Thumbnail URL not found"}
            
            # Download the thumbnail image
            img_response = requests.get(thumbnail_url, timeout=30)
            if img_response.status_code != 200:
                return {"error": f"Failed to download thumbnail: {img_response.status_code}"}
            
            # Encode image as base64
            image_data = base64.b64encode(img_response.content).decode('ascii')
            
            # Determine format from content type or URL
            content_type = img_response.headers.get("Content-Type", "")
            if "png" in content_type or thumbnail_url.endswith(".png"):
                img_format = "png"
            else:
                img_format = "jpeg"
            
            # Get additional model info for context
            model_name = data.get("name", "Unknown")
            author = data.get("user", {}).get("username", "Unknown")
            
            return {
                "success": True,
                "image_data": image_data,
                "format": img_format,
                "model_name": model_name,
                "author": author,
                "uid": uid,
                "thumbnail_width": selected_thumbnail.get("width"),
                "thumbnail_height": selected_thumbnail.get("height")
            }
            
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Check your internet connection."}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to get model preview: {str(e)}"}

    def get_sketchfab_model_license(self, uid):
        """Get normalized license/attribution details for a Sketchfab model."""
        try:
            api_key = bpy.context.scene.blendermcp_sketchfab_api_key
            if not api_key:
                return {"error": "Sketchfab API key is not configured"}

            headers = {"Authorization": f"Token {api_key}"}
            response = requests.get(
                f"https://api.sketchfab.com/v3/models/{uid}",
                headers=headers,
                timeout=30
            )

            if response.status_code == 401:
                return {"error": "Authentication failed (401). Check your API key."}
            if response.status_code == 404:
                return {"error": f"Model not found: {uid}"}
            if response.status_code != 200:
                return {"error": f"Failed to get model details: {response.status_code}"}

            data = response.json() or {}
            license_data = data.get("license") or {}
            user_data = data.get("user") or {}

            license_label = license_data.get("label", "Unknown")
            license_slug = str(license_data.get("slug", "")).lower()
            attribution_required = license_slug.startswith("cc-by") or ("attribution" in license_label.lower())

            return {
                "uid": uid,
                "name": data.get("name", "Unknown"),
                "source": "sketchfab",
                "license_code": license_data.get("fullName") or license_label,
                "license_label": license_label,
                "license_url": license_data.get("url"),
                "attribution_required": attribution_required,
                "author": user_data.get("displayName") or user_data.get("username", "Unknown"),
                "author_url": user_data.get("profileUrl"),
                "source_url": data.get("viewerUrl") or data.get("uri"),
                "commercial_use_allowed": True,
            }
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Check your internet connection."}
        except Exception as e:
            return {"error": f"Failed to get model license: {str(e)}"}

    def download_sketchfab_model(self, uid, normalize_size=False, target_size=1.0):
        """Download a model from Sketchfab by its UID
        
        Parameters:
        - uid: The unique identifier of the Sketchfab model
        - normalize_size: If True, scale the model so its largest dimension equals target_size
        - target_size: The target size in Blender units (meters) for the largest dimension
        """
        try:
            api_key = bpy.context.scene.blendermcp_sketchfab_api_key
            if not api_key:
                return {"error": "Sketchfab API key is not configured"}

            # Use proper authorization header for API key auth
            headers = {
                "Authorization": f"Token {api_key}"
            }

            # Request download URL using the exact endpoint from the documentation
            download_endpoint = f"https://api.sketchfab.com/v3/models/{uid}/download"

            response = requests.get(
                download_endpoint,
                headers=headers,
                timeout=30  # Add timeout of 30 seconds
            )

            if response.status_code == 401:
                return {"error": "Authentication failed (401). Check your API key."}

            if response.status_code != 200:
                return {"error": f"Download request failed with status code {response.status_code}"}

            data = response.json()

            # Safety check for None data
            if data is None:
                return {"error": "Received empty response from Sketchfab API for download request"}

            # Extract download URL with safety checks
            gltf_data = data.get("gltf")
            if not gltf_data:
                return {"error": "No gltf download URL available for this model. Response: " + str(data)}

            download_url = gltf_data.get("url")
            if not download_url:
                return {"error": "No download URL available for this model. Make sure the model is downloadable and you have access."}

            # Download the model (already has timeout)
            model_response = requests.get(download_url, timeout=60)  # 60 second timeout

            if model_response.status_code != 200:
                return {"error": f"Model download failed with status code {model_response.status_code}"}

            # Save to temporary file
            temp_dir = tempfile.mkdtemp()
            zip_file_path = os.path.join(temp_dir, f"{uid}.zip")

            with open(zip_file_path, "wb") as f:
                f.write(model_response.content)

            # Extract the zip file with enhanced security
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # More secure zip slip prevention
                for file_info in zip_ref.infolist():
                    # Get the path of the file
                    file_path = file_info.filename

                    # Convert directory separators to the current OS style
                    # This handles both / and \ in zip entries
                    target_path = os.path.join(temp_dir, os.path.normpath(file_path))

                    # Get absolute paths for comparison
                    abs_temp_dir = os.path.abspath(temp_dir)
                    abs_target_path = os.path.abspath(target_path)

                    # Ensure the normalized path doesn't escape the target directory
                    if not abs_target_path.startswith(abs_temp_dir):
                        with suppress(Exception):
                            shutil.rmtree(temp_dir)
                        return {"error": "Security issue: Zip contains files with path traversal attempt"}

                    # Additional explicit check for directory traversal
                    if ".." in file_path:
                        with suppress(Exception):
                            shutil.rmtree(temp_dir)
                        return {"error": "Security issue: Zip contains files with directory traversal sequence"}

                # If all files passed security checks, extract them
                zip_ref.extractall(temp_dir)

            # Find the main glTF file
            gltf_files = [f for f in os.listdir(temp_dir) if f.endswith('.gltf') or f.endswith('.glb')]

            if not gltf_files:
                with suppress(Exception):
                    shutil.rmtree(temp_dir)
                return {"error": "No glTF file found in the downloaded model"}

            main_file = os.path.join(temp_dir, gltf_files[0])

            # Import the model
            bpy.ops.import_scene.gltf(filepath=main_file)

            # Get the imported objects
            imported_objects = list(bpy.context.selected_objects)
            imported_object_names = [obj.name for obj in imported_objects]

            # Clean up temporary files
            with suppress(Exception):
                shutil.rmtree(temp_dir)

            # Find root objects (objects without parents in the imported set)
            root_objects = [obj for obj in imported_objects if obj.parent is None]

            # Helper function to recursively get all mesh children
            def get_all_mesh_children(obj):
                """Recursively collect all mesh objects in the hierarchy"""
                meshes = []
                if obj.type == 'MESH':
                    meshes.append(obj)
                for child in obj.children:
                    meshes.extend(get_all_mesh_children(child))
                return meshes

            # Collect ALL meshes from the entire hierarchy (starting from roots)
            all_meshes = []
            for obj in root_objects:
                all_meshes.extend(get_all_mesh_children(obj))
            
            if all_meshes:
                # Calculate combined world bounding box for all meshes
                all_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
                all_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
                
                for mesh_obj in all_meshes:
                    # Get world-space bounding box corners
                    for corner in mesh_obj.bound_box:
                        world_corner = mesh_obj.matrix_world @ mathutils.Vector(corner)
                        all_min.x = min(all_min.x, world_corner.x)
                        all_min.y = min(all_min.y, world_corner.y)
                        all_min.z = min(all_min.z, world_corner.z)
                        all_max.x = max(all_max.x, world_corner.x)
                        all_max.y = max(all_max.y, world_corner.y)
                        all_max.z = max(all_max.z, world_corner.z)
                
                # Calculate dimensions
                dimensions = [
                    all_max.x - all_min.x,
                    all_max.y - all_min.y,
                    all_max.z - all_min.z
                ]
                max_dimension = max(dimensions)
                
                # Apply normalization if requested
                scale_applied = 1.0
                if normalize_size and max_dimension > 0:
                    scale_factor = target_size / max_dimension
                    scale_applied = scale_factor
                    
                    # ✅ Only apply scale to ROOT objects (not children!)
                    # Child objects inherit parent's scale through matrix_world
                    for root in root_objects:
                        root.scale = (
                            root.scale.x * scale_factor,
                            root.scale.y * scale_factor,
                            root.scale.z * scale_factor
                        )
                    
                    # Update the scene to recalculate matrix_world for all objects
                    bpy.context.view_layer.update()
                    
                    # Recalculate bounding box after scaling
                    all_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
                    all_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
                    
                    for mesh_obj in all_meshes:
                        for corner in mesh_obj.bound_box:
                            world_corner = mesh_obj.matrix_world @ mathutils.Vector(corner)
                            all_min.x = min(all_min.x, world_corner.x)
                            all_min.y = min(all_min.y, world_corner.y)
                            all_min.z = min(all_min.z, world_corner.z)
                            all_max.x = max(all_max.x, world_corner.x)
                            all_max.y = max(all_max.y, world_corner.y)
                            all_max.z = max(all_max.z, world_corner.z)
                    
                    dimensions = [
                        all_max.x - all_min.x,
                        all_max.y - all_min.y,
                        all_max.z - all_min.z
                    ]
                
                world_bounding_box = [[all_min.x, all_min.y, all_min.z], [all_max.x, all_max.y, all_max.z]]
            else:
                world_bounding_box = None
                dimensions = None
                scale_applied = 1.0

            result = {
                "success": True,
                "message": "Model imported successfully",
                "imported_objects": imported_object_names
            }
            
            if world_bounding_box:
                result["world_bounding_box"] = world_bounding_box
            if dimensions:
                result["dimensions"] = [round(d, 4) for d in dimensions]
            if normalize_size:
                result["scale_applied"] = round(scale_applied, 6)
                result["normalized"] = True
            
            return result

        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Check your internet connection and try again with a simpler model."}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response from Sketchfab API: {str(e)}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to download model: {str(e)}"}
    #endregion

    #region Hunyuan3D
    def get_hunyuan3d_status(self):
        """Get the current status of Hunyuan3D integration"""
        enabled = bpy.context.scene.blendermcp_use_hunyuan3d
        hunyuan3d_mode = bpy.context.scene.blendermcp_hunyuan3d_mode
        if enabled:
            match hunyuan3d_mode:
                case "OFFICIAL_API":
                    if not bpy.context.scene.blendermcp_hunyuan3d_secret_id or not bpy.context.scene.blendermcp_hunyuan3d_secret_key:
                        return {
                            "enabled": False, 
                            "mode": hunyuan3d_mode, 
                            "message": """Hunyuan3D integration is currently enabled, but SecretId or SecretKey is not given. To enable it:
                                1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                                2. Keep the 'Use Tencent Hunyuan 3D model generation' checkbox checked
                                3. Choose the right platform and fill in the SecretId and SecretKey
                                4. Restart the connection to Claude"""
                        }
                case "LOCAL_API":
                    if not bpy.context.scene.blendermcp_hunyuan3d_api_url:
                        return {
                            "enabled": False, 
                            "mode": hunyuan3d_mode, 
                            "message": """Hunyuan3D integration is currently enabled, but API URL  is not given. To enable it:
                                1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                                2. Keep the 'Use Tencent Hunyuan 3D model generation' checkbox checked
                                3. Choose the right platform and fill in the API URL
                                4. Restart the connection to Claude"""
                        }
                case _:
                    return {
                        "enabled": False, 
                        "message": "Hunyuan3D integration is enabled and mode is not supported."
                    }
            return {
                "enabled": True, 
                "mode": hunyuan3d_mode,
                "message": "Hunyuan3D integration is enabled and ready to use."
            }
        return {
            "enabled": False, 
            "message": """Hunyuan3D integration is currently disabled. To enable it:
                        1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                        2. Check the 'Use Tencent Hunyuan 3D model generation' checkbox
                        3. Restart the connection to Claude"""
        }
    
    @staticmethod
    def get_tencent_cloud_sign_headers(
        method: str,
        path: str,
        headParams: dict,
        data: dict,
        service: str,
        region: str,
        secret_id: str,
        secret_key: str,
        host: str = None
    ):
        """Generate the signature header required for Tencent Cloud API requests headers"""
        # Generate timestamp
        timestamp = int(time.time())
        date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
        
        # If host is not provided, it is generated based on service and region.
        if not host:
            host = f"{service}.tencentcloudapi.com"
        
        endpoint = f"https://{host}"
        
        # Constructing the request body
        payload_str = json.dumps(data)
        
        # ************* Step 1: Concatenate the canonical request string *************
        canonical_uri = path
        canonical_querystring = ""
        ct = "application/json; charset=utf-8"
        canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{headParams.get('Action', '').lower()}\n"
        signed_headers = "content-type;host;x-tc-action"
        hashed_request_payload = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        
        canonical_request = (method + "\n" +
                            canonical_uri + "\n" +
                            canonical_querystring + "\n" +
                            canonical_headers + "\n" +
                            signed_headers + "\n" +
                            hashed_request_payload)

        # ************* Step 2: Construct the reception signature string *************
        credential_scope = f"{date}/{service}/tc3_request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = ("TC3-HMAC-SHA256" + "\n" +
                        str(timestamp) + "\n" +
                        credential_scope + "\n" +
                        hashed_canonical_request)

        # ************* Step 3: Calculate the signature *************
        def sign(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
        secret_service = sign(secret_date, service)
        secret_signing = sign(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, 
            string_to_sign.encode("utf-8"), 
            hashlib.sha256
        ).hexdigest()

        # ************* Step 4: Connect Authorization *************
        authorization = ("TC3-HMAC-SHA256" + " " +
                        "Credential=" + secret_id + "/" + credential_scope + ", " +
                        "SignedHeaders=" + signed_headers + ", " +
                        "Signature=" + signature)

        # Constructing request headers
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": host,
            "X-TC-Action": headParams.get("Action", ""),
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": headParams.get("Version", ""),
            "X-TC-Region": region
        }

        return headers, endpoint

    def create_hunyuan_job(self, *args, **kwargs):
        match bpy.context.scene.blendermcp_hunyuan3d_mode:
            case "OFFICIAL_API":
                return self.create_hunyuan_job_main_site(*args, **kwargs)
            case "LOCAL_API":
                return self.create_hunyuan_job_local_site(*args, **kwargs)
            case _:
                return f"Error: Unknown Hunyuan3D mode!"

    def create_hunyuan_job_main_site(
        self,
        text_prompt: str = None,
        image: str = None
    ):
        try:
            secret_id = bpy.context.scene.blendermcp_hunyuan3d_secret_id
            secret_key = bpy.context.scene.blendermcp_hunyuan3d_secret_key

            if not secret_id or not secret_key:
                return {"error": "SecretId or SecretKey is not given"}

            # Parameter verification
            if not text_prompt and not image:
                return {"error": "Prompt or Image is required"}
            if text_prompt and image:
                return {"error": "Prompt and Image cannot be provided simultaneously"}
            # Fixed parameter configuration
            service = "hunyuan"
            action = "SubmitHunyuanTo3DJob"
            version = "2023-09-01"
            region = "ap-guangzhou"

            headParams={
                "Action": action,
                "Version": version,
                "Region": region,
            }

            # Constructing request parameters
            data = {
                "Num": 1  # The current API limit is only 1
            }

            # Handling text prompts
            if text_prompt:
                if len(text_prompt) > 200:
                    return {"error": "Prompt exceeds 200 characters limit"}
                data["Prompt"] = text_prompt

            # Handling image
            if image:
                if re.match(r'^https?://', image, re.IGNORECASE) is not None:
                    data["ImageUrl"] = image
                else:
                    try:
                        # Convert to Base64 format
                        with open(image, "rb") as f:
                            image_base64 = base64.b64encode(f.read()).decode("ascii")
                        data["ImageBase64"] = image_base64
                    except Exception as e:
                        return {"error": f"Image encoding failed: {str(e)}"}
            
            # Get signed headers
            headers, endpoint = self.get_tencent_cloud_sign_headers("POST", "/", headParams, data, service, region, secret_id, secret_key)

            response = requests.post(
                endpoint,
                headers = headers,
                data = json.dumps(data)
            )

            if response.status_code == 200:
                return response.json()
            return {
                "error": f"API request failed with status {response.status_code}: {response}"
            }
        except Exception as e:
            return {"error": str(e)}

    def create_hunyuan_job_local_site(
        self,
        text_prompt: str = None,
        image: str = None):
        try:
            base_url = bpy.context.scene.blendermcp_hunyuan3d_api_url.rstrip('/')
            octree_resolution = bpy.context.scene.blendermcp_hunyuan3d_octree_resolution
            num_inference_steps = bpy.context.scene.blendermcp_hunyuan3d_num_inference_steps
            guidance_scale = bpy.context.scene.blendermcp_hunyuan3d_guidance_scale
            texture = bpy.context.scene.blendermcp_hunyuan3d_texture

            if not base_url:
                return {"error": "API URL is not given"}
            # Parameter verification
            if not text_prompt and not image:
                return {"error": "Prompt or Image is required"}

            # Constructing request parameters
            data = {
                "octree_resolution": octree_resolution,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "texture": texture,
            }

            # Handling text prompts
            if text_prompt:
                data["text"] = text_prompt

            # Handling image
            if image:
                if re.match(r'^https?://', image, re.IGNORECASE) is not None:
                    try:
                        resImg = requests.get(image)
                        resImg.raise_for_status()
                        image_base64 = base64.b64encode(resImg.content).decode("ascii")
                        data["image"] = image_base64
                    except Exception as e:
                        return {"error": f"Failed to download or encode image: {str(e)}"} 
                else:
                    try:
                        # Convert to Base64 format
                        with open(image, "rb") as f:
                            image_base64 = base64.b64encode(f.read()).decode("ascii")
                        data["image"] = image_base64
                    except Exception as e:
                        return {"error": f"Image encoding failed: {str(e)}"}

            response = requests.post(
                f"{base_url}/generate",
                json = data,
            )

            if response.status_code != 200:
                return {
                    "error": f"Generation failed: {response.text}"
                }
        
            # Decode base64 and save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as temp_file:
                temp_file.write(response.content)
                temp_file_name = temp_file.name

            # Import the GLB file in the main thread
            def import_handler():
                bpy.ops.import_scene.gltf(filepath=temp_file_name)
                os.unlink(temp_file.name)
                return None
            
            bpy.app.timers.register(import_handler)

            return {
                "status": "DONE",
                "message": "Generation and Import glb succeeded"
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"error": str(e)}
        
    
    def poll_hunyuan_job_status(self, *args, **kwargs):
        return self.poll_hunyuan_job_status_ai(*args, **kwargs)
    
    def poll_hunyuan_job_status_ai(self, job_id: str):
        """Call the job status API to get the job status"""
        print(job_id)
        try:
            secret_id = bpy.context.scene.blendermcp_hunyuan3d_secret_id
            secret_key = bpy.context.scene.blendermcp_hunyuan3d_secret_key

            if not secret_id or not secret_key:
                return {"error": "SecretId or SecretKey is not given"}
            if not job_id:
                return {"error": "JobId is required"}
            
            service = "hunyuan"
            action = "QueryHunyuanTo3DJob"
            version = "2023-09-01"
            region = "ap-guangzhou"

            headParams={
                "Action": action,
                "Version": version,
                "Region": region,
            }

            clean_job_id = job_id.removeprefix("job_")
            data = {
                "JobId": clean_job_id
            }

            headers, endpoint = self.get_tencent_cloud_sign_headers("POST", "/", headParams, data, service, region, secret_id, secret_key)

            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(data)
            )

            if response.status_code == 200:
                return response.json()
            return {
                "error": f"API request failed with status {response.status_code}: {response}"
            }
        except Exception as e:
            return {"error": str(e)}

    def import_generated_asset_hunyuan(self, *args, **kwargs):
        return self.import_generated_asset_hunyuan_ai(*args, **kwargs)
            
    def import_generated_asset_hunyuan_ai(self, name: str , zip_file_url: str):
        if not zip_file_url:
            return {"error": "Zip file not found"}
        
        # Validate URL
        if not re.match(r'^https?://', zip_file_url, re.IGNORECASE):
            return {"error": "Invalid URL format. Must start with http:// or https://"}
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="tencent_obj_")
        zip_file_path = osp.join(temp_dir, "model.zip")
        obj_file_path = osp.join(temp_dir, "model.obj")
        mtl_file_path = osp.join(temp_dir, "model.mtl")

        try:
            # Download ZIP file
            zip_response = requests.get(zip_file_url, stream=True)
            zip_response.raise_for_status()
            with open(zip_file_path, "wb") as f:
                for chunk in zip_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Unzip the ZIP
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find the .obj file (there may be multiple, assuming the main file is model.obj)
            for file in os.listdir(temp_dir):
                if file.endswith(".obj"):
                    obj_file_path = osp.join(temp_dir, file)

            if not osp.exists(obj_file_path):
                return {"succeed": False, "error": "OBJ file not found after extraction"}

            # Import obj file
            if bpy.app.version>=(4, 0, 0):
                bpy.ops.wm.obj_import(filepath=obj_file_path)
            else:
                bpy.ops.import_scene.obj(filepath=obj_file_path)

            imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not imported_objs:
                return {"succeed": False, "error": "No mesh objects imported"}

            obj = imported_objs[0]
            if name:
                obj.name = name

            result = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            }

            if obj.type == "MESH":
                bounding_box = self._get_aabb(obj)
                result["world_bounding_box"] = bounding_box

            return {"succeed": True, **result}
        except Exception as e:
            return {"succeed": False, "error": str(e)}
        finally:
            #  Clean up temporary zip and obj, save texture and mtl
            try:
                if os.path.exists(zip_file_path):
                    os.remove(zip_file_path) 
                if os.path.exists(obj_file_path):
                    os.remove(obj_file_path)
            except Exception as e:
                print(f"Failed to clean up temporary directory {temp_dir}: {e}")
    #endregion

    # ========================================================================
    # NEW DEDICATED TOOLS - Direct manipulation handlers
    # ========================================================================

    def create_object(self, type, name=None, location=None, rotation=None, scale=None):
        """Create a new object in the scene"""
        import math
        obj_type = type.upper()
        loc = location or [0, 0, 0]

        # Map type to Blender operator
        type_map = {
            'CUBE': lambda: bpy.ops.mesh.primitive_cube_add(location=loc),
            'SPHERE': lambda: bpy.ops.mesh.primitive_uv_sphere_add(location=loc),
            'UV_SPHERE': lambda: bpy.ops.mesh.primitive_uv_sphere_add(location=loc),
            'ICO_SPHERE': lambda: bpy.ops.mesh.primitive_ico_sphere_add(location=loc),
            'CYLINDER': lambda: bpy.ops.mesh.primitive_cylinder_add(location=loc),
            'CONE': lambda: bpy.ops.mesh.primitive_cone_add(location=loc),
            'TORUS': lambda: bpy.ops.mesh.primitive_torus_add(location=loc),
            'PLANE': lambda: bpy.ops.mesh.primitive_plane_add(location=loc),
            'CIRCLE': lambda: bpy.ops.mesh.primitive_circle_add(location=loc),
            'GRID': lambda: bpy.ops.mesh.primitive_grid_add(location=loc),
            'MONKEY': lambda: bpy.ops.mesh.primitive_monkey_add(location=loc),
            'EMPTY': lambda: bpy.ops.object.empty_add(location=loc),
            'CAMERA': lambda: bpy.ops.object.camera_add(location=loc),
            'LIGHT_POINT': lambda: bpy.ops.object.light_add(type='POINT', location=loc),
            'LIGHT_SUN': lambda: bpy.ops.object.light_add(type='SUN', location=loc),
            'LIGHT_SPOT': lambda: bpy.ops.object.light_add(type='SPOT', location=loc),
            'LIGHT_AREA': lambda: bpy.ops.object.light_add(type='AREA', location=loc),
            'BEZIER_CURVE': lambda: bpy.ops.curve.primitive_bezier_curve_add(location=loc),
            'NURBS_CURVE': lambda: bpy.ops.curve.primitive_nurbs_curve_add(location=loc),
            'TEXT': lambda: bpy.ops.object.text_add(location=loc),
        }

        creator = type_map.get(obj_type)
        if not creator:
            return {"error": f"Unknown object type: {obj_type}. Available: {', '.join(type_map.keys())}"}

        creator()
        obj = bpy.context.active_object

        if name:
            obj.name = name
        if rotation:
            obj.rotation_euler = [math.radians(r) for r in rotation]
        if scale:
            obj.scale = scale

        result = {
            "name": obj.name,
            "type": obj.type,
            "location": list(obj.location),
            "created": True
        }
        if obj.type == 'MESH':
            result["world_bounding_box"] = self._get_aabb(obj)
        return result

    def delete_object(self, name, delete_children=False):
        """Delete an object from the scene"""
        obj = bpy.data.objects.get(name)
        if not obj:
            return {"error": f"Object '{name}' not found"}

        objects_to_delete = [obj]
        if delete_children:
            objects_to_delete.extend(obj.children_recursive)

        deleted_names = [o.name for o in objects_to_delete]
        for o in objects_to_delete:
            bpy.data.objects.remove(o, do_unlink=True)

        return {"deleted": deleted_names, "count": len(deleted_names)}

    def set_transform(self, object_name, location=None, rotation=None, scale=None):
        """Set the transform of an object"""
        import math
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": f"Object '{object_name}' not found"}

        if location is not None:
            obj.location = location
        if rotation is not None:
            obj.rotation_euler = [math.radians(r) for r in rotation]
        if scale is not None:
            obj.scale = scale

        return {
            "name": obj.name,
            "location": list(obj.location),
            "rotation_euler": [round(math.degrees(r), 2) for r in obj.rotation_euler],
            "scale": list(obj.scale),
            "updated": True
        }

    def apply_transforms(self, object_name, location=True, rotation=True, scale=True):
        """Apply transforms on an object"""
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": f"Object '{object_name}' not found"}

        # Select only this object
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)

        return {"name": obj.name, "applied": True}

    def add_modifier(self, object_name, modifier_type, modifier_name=None, params=None):
        """Add a modifier to an object"""
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": f"Object '{object_name}' not found"}

        mod = obj.modifiers.new(name=modifier_name or modifier_type, type=modifier_type)

        # Apply optional parameters
        if params:
            for key, value in params.items():
                try:
                    setattr(mod, key, value)
                except Exception as e:
                    print(f"Warning: Could not set modifier param {key}={value}: {e}")

        return {
            "name": mod.name,
            "type": mod.type,
            "object": obj.name,
            "added": True
        }

    def remove_modifier(self, object_name, modifier_name):
        """Remove a modifier from an object"""
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": f"Object '{object_name}' not found"}

        mod = obj.modifiers.get(modifier_name)
        if not mod:
            return {"error": f"Modifier '{modifier_name}' not found on '{object_name}'"}

        obj.modifiers.remove(mod)
        return {"removed": modifier_name, "object": obj.name}

    def apply_modifier(self, object_name, modifier_name):
        """Apply a modifier permanently"""
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": f"Object '{object_name}' not found"}

        mod = obj.modifiers.get(modifier_name)
        if not mod:
            return {"error": f"Modifier '{modifier_name}' not found on '{object_name}'"}

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier_name)

        return {"applied": modifier_name, "object": obj.name}

    def create_material(self, name, base_color=None, metallic=0.0, roughness=0.5,
                        emission_color=None, emission_strength=0.0, alpha=1.0, use_nodes=True):
        """Create a new material with Principled BSDF"""
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = use_nodes

        if use_nodes:
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                if base_color:
                    bsdf.inputs['Base Color'].default_value = base_color if len(base_color) == 4 else base_color + [1.0]
                bsdf.inputs['Metallic'].default_value = metallic
                bsdf.inputs['Roughness'].default_value = roughness
                if emission_color:
                    # Handle Blender 3.x ('Emission') vs 4.x ('Emission Color')
                    emission_input = bsdf.inputs.get('Emission Color') or bsdf.inputs.get('Emission')
                    if emission_input:
                        emission_input.default_value = emission_color if len(emission_color) == 4 else emission_color + [1.0]
                # Handle Blender 3.x vs 4.x emission strength
                strength_input = bsdf.inputs.get('Emission Strength')
                if strength_input:
                    strength_input.default_value = emission_strength
                bsdf.inputs['Alpha'].default_value = alpha

                if alpha < 1.0:
                    if hasattr(mat, 'blend_method'):
                        mat.blend_method = 'BLEND'

        return {"name": mat.name, "created": True}

    def assign_material(self, object_name, material_name):
        """Assign a material to an object"""
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": f"Object '{object_name}' not found"}

        mat = bpy.data.materials.get(material_name)
        if not mat:
            return {"error": f"Material '{material_name}' not found"}

        if obj.data and hasattr(obj.data, 'materials'):
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
            return {"object": obj.name, "material": mat.name, "assigned": True}
        else:
            return {"error": f"Object '{object_name}' does not support materials"}

    def render_image(self, filepath=None, resolution_x=1920, resolution_y=1080,
                     engine=None, samples=None):
        """Render the current scene to an image"""
        scene = bpy.context.scene

        # Set render settings
        scene.render.resolution_x = resolution_x
        scene.render.resolution_y = resolution_y

        if engine:
            scene.render.engine = engine

        if samples:
            if scene.render.engine == 'CYCLES':
                scene.cycles.samples = samples
            elif scene.render.engine == 'BLENDER_EEVEE_NEXT' or scene.render.engine == 'BLENDER_EEVEE':
                scene.eevee.taa_render_samples = samples

        if filepath:
            scene.render.filepath = filepath
        elif not scene.render.filepath:
            scene.render.filepath = "//render.png"

        bpy.ops.render.render(write_still=True)

        return {
            "rendered": True,
            "filepath": bpy.path.abspath(scene.render.filepath),
            "resolution": [resolution_x, resolution_y],
            "engine": scene.render.engine
        }

    def set_camera(self, name=None, location=None, rotation=None,
                   focal_length=None, look_at=None, set_active=True):
        """Configure a camera"""
        import math
        cam_name = name or "Camera"
        cam_obj = bpy.data.objects.get(cam_name)

        if not cam_obj:
            bpy.ops.object.camera_add()
            cam_obj = bpy.context.active_object
            cam_obj.name = cam_name

        if location:
            cam_obj.location = location

        if look_at:
            direction = mathutils.Vector(look_at) - cam_obj.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            cam_obj.rotation_euler = rot_quat.to_euler()
        elif rotation:
            cam_obj.rotation_euler = [math.radians(r) for r in rotation]

        if focal_length and cam_obj.data:
            cam_obj.data.lens = focal_length

        if set_active:
            bpy.context.scene.camera = cam_obj

        return {
            "name": cam_obj.name,
            "location": list(cam_obj.location),
            "rotation": [round(math.degrees(r), 2) for r in cam_obj.rotation_euler],
            "focal_length": cam_obj.data.lens if cam_obj.data else None,
            "is_active": bpy.context.scene.camera == cam_obj
        }

    def add_light(self, type='POINT', name=None, location=None, rotation=None,
                  energy=1000.0, color=None, size=None):
        """Add a light to the scene"""
        import math
        loc = location or [0, 0, 0]
        bpy.ops.object.light_add(type=type, location=loc)
        light_obj = bpy.context.active_object

        if name:
            light_obj.name = name
        if rotation:
            light_obj.rotation_euler = [math.radians(r) for r in rotation]

        light_obj.data.energy = energy
        if color:
            light_obj.data.color = color[:3]
        if size is not None:
            if hasattr(light_obj.data, 'shadow_soft_size'):
                light_obj.data.shadow_soft_size = size
            elif hasattr(light_obj.data, 'size'):
                light_obj.data.size = size

        return {
            "name": light_obj.name,
            "type": type,
            "location": list(light_obj.location),
            "energy": energy,
            "created": True
        }

    def manage_collection(self, action, name=None, object_name=None, parent_name=None):
        """Manage collections"""
        if action == "list":
            collections = []
            def collect_info(col, depth=0):
                collections.append({
                    "name": col.name,
                    "depth": depth,
                    "objects": [obj.name for obj in col.objects],
                    "children": [c.name for c in col.children]
                })
                for child in col.children:
                    collect_info(child, depth + 1)
            collect_info(bpy.context.scene.collection)
            return {"collections": collections}

        elif action == "create":
            if not name:
                return {"error": "Collection name is required"}
            new_col = bpy.data.collections.new(name)
            parent = None
            if parent_name:
                parent = bpy.data.collections.get(parent_name)
            if parent:
                parent.children.link(new_col)
            else:
                bpy.context.scene.collection.children.link(new_col)
            return {"name": new_col.name, "created": True}

        elif action == "move":
            if not name or not object_name:
                return {"error": "Both 'name' (target collection) and 'object_name' are required"}
            obj = bpy.data.objects.get(object_name)
            if not obj:
                return {"error": f"Object '{object_name}' not found"}
            target = bpy.data.collections.get(name)
            if not target:
                return {"error": f"Collection '{name}' not found"}
            # Unlink from all current collections
            for col in obj.users_collection:
                col.objects.unlink(obj)
            target.objects.link(obj)
            return {"object": obj.name, "collection": target.name, "moved": True}

        elif action == "delete":
            if not name:
                return {"error": "Collection name is required"}
            col = bpy.data.collections.get(name)
            if not col:
                return {"error": f"Collection '{name}' not found"}
            bpy.data.collections.remove(col)
            return {"deleted": name}

        return {"error": f"Unknown action: {action}"}

    def set_keyframe(self, object_name, frame, data_path=None,
                     location=None, rotation=None, scale=None, value=None):
        """Insert a keyframe for animation"""
        import math
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": f"Object '{object_name}' not found"}

        bpy.context.scene.frame_set(frame)
        keyed = []

        if location:
            obj.location = location
            obj.keyframe_insert(data_path='location', frame=frame)
            keyed.append('location')

        if rotation:
            obj.rotation_euler = [math.radians(r) for r in rotation]
            obj.keyframe_insert(data_path='rotation_euler', frame=frame)
            keyed.append('rotation_euler')

        if scale:
            obj.scale = scale
            obj.keyframe_insert(data_path='scale', frame=frame)
            keyed.append('scale')

        if data_path and not any([location, rotation, scale]):
            if value is not None:
                # Try setting the value via the data path
                try:
                    exec(f"obj.{data_path} = {value}")
                except:
                    pass
            obj.keyframe_insert(data_path=data_path, frame=frame)
            keyed.append(data_path)

        return {
            "object": obj.name,
            "frame": frame,
            "keyframed": keyed
        }

    def export_scene(self, filepath, format='GLTF', selected_only=False):
        """Export the scene to a file"""
        fmt = format.upper()

        # Select/deselect based on selected_only
        if not selected_only:
            bpy.ops.object.select_all(action='SELECT')

        export_map = {
            'GLTF': lambda: bpy.ops.export_scene.gltf(
                filepath=filepath, use_selection=selected_only, export_format='GLTF_SEPARATE'),
            'GLB': lambda: bpy.ops.export_scene.gltf(
                filepath=filepath, use_selection=selected_only, export_format='GLB'),
            'FBX': lambda: bpy.ops.export_scene.fbx(
                filepath=filepath, use_selection=selected_only),
            'OBJ': lambda: bpy.ops.wm.obj_export(
                filepath=filepath, export_selected_objects=selected_only) if hasattr(bpy.ops.wm, 'obj_export') else bpy.ops.export_scene.obj(
                filepath=filepath, use_selection=selected_only),
            'STL': lambda: bpy.ops.export_mesh.stl(
                filepath=filepath, use_selection=selected_only),
            'PLY': lambda: bpy.ops.export_mesh.ply(
                filepath=filepath, use_selection=selected_only),
            'USD': lambda: bpy.ops.wm.usd_export(
                filepath=filepath, selected_objects_only=selected_only),
            'ABC': lambda: bpy.ops.wm.alembic_export(
                filepath=filepath, selected=selected_only),
        }

        exporter = export_map.get(fmt)
        if not exporter:
            return {"error": f"Unknown format: {fmt}. Available: {', '.join(export_map.keys())}"}

        try:
            exporter()
        except Exception as e:
            return {"error": f"Export failed for format {fmt}: {str(e)}"}

        return {"exported": True, "filepath": filepath, "format": fmt}

    def duplicate_object(self, name, new_name=None, linked=False):
        """Duplicate an object in the scene"""
        obj = bpy.data.objects.get(name)
        if not obj:
            return {"error": f"Object '{name}' not found"}

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        if linked:
            bpy.ops.object.duplicate_move_linked()
        else:
            bpy.ops.object.duplicate_move()

        new_obj = bpy.context.active_object
        if new_name:
            new_obj.name = new_name

        return {
            "original": name,
            "duplicate": new_obj.name,
            "linked": linked,
            "location": list(new_obj.location),
            "created": True
        }

    def join_objects(self, target_name, source_names):
        """Join multiple objects into one"""
        target = bpy.data.objects.get(target_name)
        if not target:
            return {"error": f"Target object '{target_name}' not found"}

        bpy.ops.object.select_all(action='DESELECT')
        missing = []
        joined = []
        for sn in source_names:
            src = bpy.data.objects.get(sn)
            if src:
                src.select_set(True)
                joined.append(sn)
            else:
                missing.append(sn)

        target.select_set(True)
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.join()

        result = {
            "result_object": target.name,
            "joined": joined,
            "joined_count": len(joined) + 1,
        }
        if missing:
            result["missing_objects"] = missing
        return result

    def set_parent(self, child_name, parent_name=None, keep_transform=True):
        """Set or clear parent-child relationship"""
        child = bpy.data.objects.get(child_name)
        if not child:
            return {"error": f"Child object '{child_name}' not found"}

        bpy.ops.object.select_all(action='DESELECT')

        if parent_name:
            # Set parent
            parent = bpy.data.objects.get(parent_name)
            if not parent:
                return {"error": f"Parent object '{parent_name}' not found"}

            child.select_set(True)
            parent.select_set(True)
            bpy.context.view_layer.objects.active = parent

            if keep_transform:
                bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
            else:
                bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)

            return {
                "child": child.name,
                "parent": parent.name,
                "keep_transform": keep_transform,
                "set": True
            }
        else:
            # Clear parent
            child.select_set(True)
            bpy.context.view_layer.objects.active = child
            if keep_transform:
                bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            else:
                bpy.ops.object.parent_clear(type='CLEAR')

            return {
                "child": child.name,
                "parent": None,
                "cleared": True
            }

    def select_objects(self, object_names=None, clear=True, set_active=None):
        """Select objects by name and optionally set the active object."""
        if clear:
            bpy.ops.object.select_all(action='DESELECT')

        missing = []
        selected = []

        if object_names:
            for name in object_names:
                obj = bpy.data.objects.get(name)
                if obj:
                    obj.select_set(True)
                    selected.append(obj.name)
                else:
                    missing.append(name)
        else:
            selected = [obj.name for obj in bpy.context.selected_objects]

        if set_active:
            active_obj = bpy.data.objects.get(set_active)
            if active_obj and active_obj.select_get():
                bpy.context.view_layer.objects.active = active_obj
            elif active_obj and not active_obj.select_get():
                active_obj.select_set(True)
                bpy.context.view_layer.objects.active = active_obj

        active_name = bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else None

        result = {
            "selected": [obj.name for obj in bpy.context.selected_objects],
            "active": active_name,
        }
        if missing:
            result["missing"] = missing
        return result

    def frame_control(self, action="get", frame=None, start=None, end=None, fps=None):
        """Get/set timeline frame state in a deterministic way."""
        scene = bpy.context.scene
        action = (action or "get").lower()

        if action == "get":
            pass
        elif action == "set":
            if frame is None:
                return {"error": "'frame' is required when action='set'"}
            scene.frame_set(int(frame))
        elif action == "range":
            if start is not None:
                scene.frame_start = int(start)
            if end is not None:
                scene.frame_end = int(end)
            if scene.frame_end < scene.frame_start:
                return {"error": "Invalid frame range: end must be >= start"}
        elif action == "fps":
            if fps is None:
                return {"error": "'fps' is required when action='fps'"}
            scene.render.fps = int(fps)
        else:
            return {"error": "Unknown action. Use: get, set, range, fps"}

        return {
            "frame_current": scene.frame_current,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "fps": scene.render.fps,
        }

    def save_blend_file(self, filepath=None, pack_resources=False):
        """Save current .blend file, optionally packing external resources."""
        try:
            if pack_resources:
                bpy.ops.file.pack_all()

            if filepath:
                bpy.ops.wm.save_as_mainfile(filepath=filepath)
            else:
                bpy.ops.wm.save_mainfile()

            return {
                "saved": True,
                "filepath": bpy.data.filepath,
                "packed_resources": bool(pack_resources),
            }
        except Exception as e:
            return {"error": f"Failed to save .blend file: {str(e)}"}

    def open_blend_file(self, filepath):
        """Open a .blend file from disk."""
        try:
            if not filepath:
                return {"error": "'filepath' is required"}
            if not os.path.exists(filepath):
                return {"error": f"File not found: {filepath}"}
            bpy.ops.wm.open_mainfile(filepath=filepath)
            return {"opened": True, "filepath": bpy.data.filepath}
        except Exception as e:
            return {"error": f"Failed to open .blend file: {str(e)}"}

    def import_file(self, filepath):
        """Import a file into the current scene based on extension."""
        if not filepath:
            return {"error": "'filepath' is required"}
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}

        ext = os.path.splitext(filepath)[1].lower()
        existing_objects = set(bpy.data.objects)

        try:
            if ext in [".glb", ".gltf"]:
                bpy.ops.import_scene.gltf(filepath=filepath)
            elif ext == ".fbx":
                bpy.ops.import_scene.fbx(filepath=filepath)
            elif ext == ".obj":
                if hasattr(bpy.ops.wm, 'obj_import'):
                    bpy.ops.wm.obj_import(filepath=filepath)
                else:
                    bpy.ops.import_scene.obj(filepath=filepath)
            elif ext == ".stl":
                bpy.ops.import_mesh.stl(filepath=filepath)
            elif ext == ".ply":
                bpy.ops.import_mesh.ply(filepath=filepath)
            elif ext in [".dae"]:
                bpy.ops.wm.collada_import(filepath=filepath)
            elif ext in [".abc"]:
                bpy.ops.wm.alembic_import(filepath=filepath)
            elif ext in [".usd", ".usda", ".usdc"]:
                bpy.ops.wm.usd_import(filepath=filepath)
            else:
                return {"error": f"Unsupported import format: {ext}"}

            imported = list(set(bpy.data.objects) - existing_objects)
            imported_names = [obj.name for obj in imported]
            return {
                "imported": True,
                "filepath": filepath,
                "count": len(imported_names),
                "objects": imported_names,
            }
        except Exception as e:
            return {"error": f"Failed to import file: {str(e)}"}

    def search_blender_docs(self, query, category="all", max_results=5):
        """RAG pipeline for searching Blender Python API documentation.

        Three-tier scoring: filename match (3x), heading match (2x), body match (1x).
        Multi-term tokenization for flexible matching.
        Section-aware snippet extraction returns complete function/class docs.
        """
        import glob
        import re

        # ── 1. Locate docs directory ──────────────────────────────────
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        env_docs_dir = os.getenv("BLENDER_DOCS_PATH", "").strip()
        cwd = os.getcwd()
        possible_paths = [
            env_docs_dir,
            os.path.join(addon_dir, "blender_docs_md"),
            os.path.join(addon_dir, "..", "blender_docs_md"),
            os.path.join(addon_dir, "..", "..", "blender_docs_md"),
            os.path.join(cwd, "blender_docs_md"),
            os.path.join(cwd, "..", "blender_docs_md"),
        ]

        docs_dir = None
        for p in possible_paths:
            if p and os.path.isdir(p):
                docs_dir = os.path.abspath(p)
                break

        if not docs_dir:
            return {
                "error": (
                    "Blender docs directory not found. Set BLENDER_DOCS_PATH to your docs folder, "
                    "or place 'blender_docs_md' near addon.py / working directory. "
                    f"Searched near addon: {addon_dir}, cwd: {cwd}"
                )
            }

        # ── 2. Category → subdirectory mapping ────────────────────────
        category_map = {
            "bpy_ops": ["bpy_ops"],
            "bpy_types": ["bpy_types"],
            "guides": ["guides"],
            "bmesh": ["bmesh"],
            "mathutils": ["mathutils"],
            "bpy_core": ["bpy_core"],
            "bpy_extras": ["bpy_extras"],
            "bpy_app": ["bpy_app"],
            "gpu": ["gpu"],
            "freestyle": ["freestyle"],
            "other": ["other"],
            "all": ["bpy_ops", "bpy_types", "guides", "bmesh", "mathutils",
                    "bpy_core", "bpy_extras", "bpy_app", "gpu", "freestyle", "other"],
        }
        search_dirs = category_map.get(category.lower(), category_map["all"])

        # ── 3. Tokenize query ─────────────────────────────────────────
        # Split on whitespace, dots, underscores → lowercase tokens
        raw_tokens = re.split(r'[\s._]+', query.strip())
        tokens = [t.lower() for t in raw_tokens if len(t) >= 2]
        query_lower = query.lower().strip()

        if not tokens:
            return {"message": "Query too short. Try terms like 'mesh', 'modifier', 'material', 'camera'.", "results": []}

        # ── 4. Score all candidate files ──────────────────────────────
        scored_files = []  # list of (score, filepath, subdir)

        for subdir in search_dirs:
            search_path = os.path.join(docs_dir, subdir)
            if not os.path.exists(search_path):
                continue

            for md_file in glob.glob(os.path.join(search_path, "**", "*.md"), recursive=True):
                filename = os.path.basename(md_file).lower()
                score = 0.0

                # Tier 1: Filename matching (weight 3x)
                # Exact full‑query match in filename
                if query_lower in filename:
                    score += 30.0
                # Each token in filename
                for tok in tokens:
                    if tok in filename:
                        score += 3.0

                # Quick-read first 500 bytes for heading sniff
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        head_text = f.read(2000)
                except Exception:
                    continue

                head_lower = head_text.lower()

                # Tier 2: Heading matching (weight 2x)
                # Extract markdown headings from the first 2KB
                headings = re.findall(r'^#{1,3}\s+(.+)$', head_text, re.MULTILINE)
                headings_lower = ' '.join(h.lower() for h in headings)
                if query_lower in headings_lower:
                    score += 20.0
                for tok in tokens:
                    if tok in headings_lower:
                        score += 2.0

                # Tier 3: Body presence (weight 1x) — just check first 2KB
                for tok in tokens:
                    if tok in head_lower:
                        score += 1.0

                if score > 0:
                    scored_files.append((score, md_file, subdir))

        # Sort by score descending
        scored_files.sort(key=lambda x: -x[0])

        if not scored_files:
            return {
                "message": f"No documentation found for '{query}'",
                "suggestion": "Try broader terms like 'mesh', 'modifier', 'material', 'camera', 'light', 'constraint', 'particle'.",
                "results": []
            }

        # ── 5. Extract relevant snippets from top files ───────────────
        max_results = min(max(1, max_results), 10)
        results = []

        for score, md_file, subdir in scored_files[:max_results]:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue

            lines = content.split('\n')
            content_lower = content.lower()

            # Find matching sections — look for the best section that contains query tokens
            snippet = self._extract_best_section(lines, tokens, query_lower)

            results.append({
                "file": os.path.basename(md_file),
                "category": subdir,
                "relevance_score": round(score, 1),
                "snippet": snippet[:6000]  # Cap at 6KB per result
            })

        return {
            "results": results,
            "count": len(results),
            "total_candidates": len(scored_files),
            "query_tokens": tokens
        }

    def _extract_best_section(self, lines, tokens, query_lower):
        """Extract the most relevant section from a document.

        Strategy: find the best matching heading-delimited section,
        or fall back to context around the best-matching line.
        """
        import re

        # Build a list of sections: (heading_line_idx, heading_text)
        sections = []
        for i, line in enumerate(lines):
            if re.match(r'^#{1,4}\s+', line):
                sections.append(i)

        # If no headings, treat entire file as one section
        if not sections:
            # Find best matching line and return context
            return self._context_around_best_match(lines, tokens, query_lower, context_before=3, context_after=20)

        # Score each section
        best_section_idx = 0
        best_score = -1

        for s_idx, start_line in enumerate(sections):
            end_line = sections[s_idx + 1] if s_idx + 1 < len(sections) else len(lines)
            section_text = '\n'.join(lines[start_line:end_line]).lower()

            score = 0
            if query_lower in section_text:
                score += 10
            for tok in tokens:
                score += section_text.count(tok)

            if score > best_score:
                best_score = score
                best_section_idx = s_idx

        # Extract best section
        start = sections[best_section_idx]
        end = sections[best_section_idx + 1] if best_section_idx + 1 < len(sections) else len(lines)

        # Cap section length at 120 lines — if longer, extract sub-context
        section_lines = lines[start:end]
        if len(section_lines) > 120:
            # Find best match within section and extract focused context
            sub_snippet = self._context_around_best_match(section_lines, tokens, query_lower, context_before=5, context_after=30)
            return f"{lines[start]}\n\n{sub_snippet}"

        return '\n'.join(section_lines)

    def _context_around_best_match(self, lines, tokens, query_lower, context_before=3, context_after=20):
        """Find the best matching line and return surrounding context."""
        best_line_idx = 0
        best_score = -1

        for i, line in enumerate(lines):
            line_lower = line.lower()
            score = 0
            if query_lower in line_lower:
                score += 10
            for tok in tokens:
                if tok in line_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_line_idx = i

        start = max(0, best_line_idx - context_before)
        end = min(len(lines), best_line_idx + context_after)
        return '\n'.join(lines[start:end])

# Blender Addon Preferences
class BLENDERMCP_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        layout.label(text="Blender MCP v2.0.0 — by Aymen Mabrouk", icon='INFO')
        layout.label(text="Connect any AI assistant to Blender via MCP.")

# Blender UI Panel
class BLENDERMCP_PT_Panel(bpy.types.Panel):
    bl_label = "Blender MCP"
    bl_idname = "BLENDERMCP_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderMCP'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "blendermcp_port")
        layout.prop(scene, "blendermcp_use_polyhaven", text="Use assets from Poly Haven")

        layout.prop(scene, "blendermcp_use_hyper3d", text="Use Hyper3D Rodin 3D model generation")
        if scene.blendermcp_use_hyper3d:
            layout.prop(scene, "blendermcp_hyper3d_mode", text="Rodin Mode")
            layout.prop(scene, "blendermcp_hyper3d_api_key", text="API Key")
            layout.operator("blendermcp.set_hyper3d_free_trial_api_key", text="Set Free Trial API Key")

        layout.prop(scene, "blendermcp_use_sketchfab", text="Use assets from Sketchfab")
        if scene.blendermcp_use_sketchfab:
            layout.prop(scene, "blendermcp_sketchfab_api_key", text="API Key")

        layout.prop(scene, "blendermcp_use_polypizza", text="Use assets from Poly Pizza")
        if scene.blendermcp_use_polypizza:
            layout.prop(scene, "blendermcp_polypizza_api_key", text="API Key")

        layout.prop(scene, "blendermcp_use_hunyuan3d", text="Use Tencent Hunyuan 3D model generation")
        if scene.blendermcp_use_hunyuan3d:
            layout.prop(scene, "blendermcp_hunyuan3d_mode", text="Hunyuan3D Mode")
            if scene.blendermcp_hunyuan3d_mode == 'OFFICIAL_API':
                layout.prop(scene, "blendermcp_hunyuan3d_secret_id", text="SecretId")
                layout.prop(scene, "blendermcp_hunyuan3d_secret_key", text="SecretKey")
            if scene.blendermcp_hunyuan3d_mode == 'LOCAL_API':
                layout.prop(scene, "blendermcp_hunyuan3d_api_url", text="API URL")
                layout.prop(scene, "blendermcp_hunyuan3d_octree_resolution", text="Octree Resolution")
                layout.prop(scene, "blendermcp_hunyuan3d_num_inference_steps", text="Number of Inference Steps")
                layout.prop(scene, "blendermcp_hunyuan3d_guidance_scale", text="Guidance Scale")
                layout.prop(scene, "blendermcp_hunyuan3d_texture", text="Generate Texture")
        
        if not scene.blendermcp_server_running:
            layout.operator("blendermcp.start_server", text="Connect to MCP server")
        else:
            layout.operator("blendermcp.stop_server", text="Disconnect from MCP server")
            layout.label(text=f"Running on port {scene.blendermcp_port}")

# Operator to set Hyper3D API Key
class BLENDERMCP_OT_SetFreeTrialHyper3DAPIKey(bpy.types.Operator):
    bl_idname = "blendermcp.set_hyper3d_free_trial_api_key"
    bl_label = "Set Free Trial API Key"

    def execute(self, context):
        context.scene.blendermcp_hyper3d_api_key = RODIN_FREE_TRIAL_KEY
        context.scene.blendermcp_hyper3d_mode = 'MAIN_SITE'
        self.report({'INFO'}, "API Key set successfully!")
        return {'FINISHED'}

# Operator to start the server
class BLENDERMCP_OT_StartServer(bpy.types.Operator):
    bl_idname = "blendermcp.start_server"
    bl_label = "Connect to MCP Server"
    bl_description = "Start the BlenderMCP server to connect with an AI assistant"

    def execute(self, context):
        scene = context.scene

        # Create a new server instance
        if not hasattr(bpy.types, "blendermcp_server") or not bpy.types.blendermcp_server:
            bpy.types.blendermcp_server = BlenderMCPServer(port=scene.blendermcp_port)

        # Start the server
        bpy.types.blendermcp_server.start()
        scene.blendermcp_server_running = True

        return {'FINISHED'}

# Operator to stop the server
class BLENDERMCP_OT_StopServer(bpy.types.Operator):
    bl_idname = "blendermcp.stop_server"
    bl_label = "Disconnect from MCP Server"
    bl_description = "Stop the BlenderMCP server connection"

    def execute(self, context):
        scene = context.scene

        # Stop the server if it exists
        if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
            bpy.types.blendermcp_server.stop()
            del bpy.types.blendermcp_server

        scene.blendermcp_server_running = False

        return {'FINISHED'}

# Operator to open Terms and Conditions
class BLENDERMCP_OT_OpenTerms(bpy.types.Operator):
    bl_idname = "blendermcp.open_terms"
    bl_label = "View Terms and Conditions"
    bl_description = "Open the Terms and Conditions document"

    def execute(self, context):
        # Open the Terms and Conditions on GitHub
        terms_url = "https://github.com/aymenmabrouk/blender-mcp"
        try:
            import webbrowser
            webbrowser.open(terms_url)
            self.report({'INFO'}, "Terms and Conditions opened in browser")
        except Exception as e:
            self.report({'ERROR'}, f"Could not open Terms and Conditions: {str(e)}")
        
        return {'FINISHED'}

# Registration functions
def register():
    bpy.types.Scene.blendermcp_port = IntProperty(
        name="Port",
        description="Port for the BlenderMCP server",
        default=9876,
        min=1024,
        max=65535
    )

    bpy.types.Scene.blendermcp_server_running = bpy.props.BoolProperty(
        name="Server Running",
        default=False
    )

    bpy.types.Scene.blendermcp_use_polyhaven = bpy.props.BoolProperty(
        name="Use Poly Haven",
        description="Enable Poly Haven asset integration",
        default=False
    )

    bpy.types.Scene.blendermcp_use_hyper3d = bpy.props.BoolProperty(
        name="Use Hyper3D Rodin",
        description="Enable Hyper3D Rodin generatino integration",
        default=False
    )

    bpy.types.Scene.blendermcp_hyper3d_mode = bpy.props.EnumProperty(
        name="Rodin Mode",
        description="Choose the platform used to call Rodin APIs",
        items=[
            ("MAIN_SITE", "hyper3d.ai", "hyper3d.ai"),
            ("FAL_AI", "fal.ai", "fal.ai"),
        ],
        default="MAIN_SITE"
    )

    bpy.types.Scene.blendermcp_hyper3d_api_key = bpy.props.StringProperty(
        name="Hyper3D API Key",
        subtype="PASSWORD",
        description="API Key provided by Hyper3D",
        default=""
    )

    bpy.types.Scene.blendermcp_use_hunyuan3d = bpy.props.BoolProperty(
        name="Use Hunyuan 3D",
        description="Enable Hunyuan asset integration",
        default=False
    )

    bpy.types.Scene.blendermcp_hunyuan3d_mode = bpy.props.EnumProperty(
        name="Hunyuan3D Mode",
        description="Choose a local or official APIs",
        items=[
            ("LOCAL_API", "local api", "local api"),
            ("OFFICIAL_API", "official api", "official api"),
        ],
        default="LOCAL_API"
    )

    bpy.types.Scene.blendermcp_hunyuan3d_secret_id = bpy.props.StringProperty(
        name="Hunyuan 3D SecretId",
        description="SecretId provided by Hunyuan 3D",
        default=""
    )

    bpy.types.Scene.blendermcp_hunyuan3d_secret_key = bpy.props.StringProperty(
        name="Hunyuan 3D SecretKey",
        subtype="PASSWORD",
        description="SecretKey provided by Hunyuan 3D",
        default=""
    )

    bpy.types.Scene.blendermcp_hunyuan3d_api_url = bpy.props.StringProperty(
        name="API URL",
        description="URL of the Hunyuan 3D API service",
        default="http://localhost:8081"
    )

    bpy.types.Scene.blendermcp_hunyuan3d_octree_resolution = bpy.props.IntProperty(
        name="Octree Resolution",
        description="Octree resolution for the 3D generation",
        default=256,
        min=128,
        max=512,
    )

    bpy.types.Scene.blendermcp_hunyuan3d_num_inference_steps = bpy.props.IntProperty(
        name="Number of Inference Steps",
        description="Number of inference steps for the 3D generation",
        default=20,
        min=20,
        max=50,
    )

    bpy.types.Scene.blendermcp_hunyuan3d_guidance_scale = bpy.props.FloatProperty(
        name="Guidance Scale",
        description="Guidance scale for the 3D generation",
        default=5.5,
        min=1.0,
        max=10.0,
    )

    bpy.types.Scene.blendermcp_hunyuan3d_texture = bpy.props.BoolProperty(
        name="Generate Texture",
        description="Whether to generate texture for the 3D model",
        default=False,
    )
    
    bpy.types.Scene.blendermcp_use_sketchfab = bpy.props.BoolProperty(
        name="Use Sketchfab",
        description="Enable Sketchfab asset integration",
        default=False
    )

    bpy.types.Scene.blendermcp_sketchfab_api_key = bpy.props.StringProperty(
        name="Sketchfab API Key",
        subtype="PASSWORD",
        description="API Key provided by Sketchfab",
        default=""
    )

    bpy.types.Scene.blendermcp_use_polypizza = bpy.props.BoolProperty(
        name="Use Poly Pizza",
        description="Enable Poly Pizza asset integration",
        default=False
    )

    bpy.types.Scene.blendermcp_polypizza_api_key = bpy.props.StringProperty(
        name="Poly Pizza API Key",
        subtype="PASSWORD",
        description="API Key provided by Poly Pizza",
        default=""
    )

    # Register preferences class
    bpy.utils.register_class(BLENDERMCP_AddonPreferences)

    bpy.utils.register_class(BLENDERMCP_PT_Panel)
    bpy.utils.register_class(BLENDERMCP_OT_SetFreeTrialHyper3DAPIKey)
    bpy.utils.register_class(BLENDERMCP_OT_StartServer)
    bpy.utils.register_class(BLENDERMCP_OT_StopServer)
    bpy.utils.register_class(BLENDERMCP_OT_OpenTerms)

    print("BlenderMCP addon registered")

def unregister():
    # Stop the server if it's running
    if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
        bpy.types.blendermcp_server.stop()
        del bpy.types.blendermcp_server

    bpy.utils.unregister_class(BLENDERMCP_PT_Panel)
    bpy.utils.unregister_class(BLENDERMCP_OT_SetFreeTrialHyper3DAPIKey)
    bpy.utils.unregister_class(BLENDERMCP_OT_StartServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_StopServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_OpenTerms)
    bpy.utils.unregister_class(BLENDERMCP_AddonPreferences)

    del bpy.types.Scene.blendermcp_port
    del bpy.types.Scene.blendermcp_server_running
    del bpy.types.Scene.blendermcp_use_polyhaven
    del bpy.types.Scene.blendermcp_use_hyper3d
    del bpy.types.Scene.blendermcp_hyper3d_mode
    del bpy.types.Scene.blendermcp_hyper3d_api_key
    del bpy.types.Scene.blendermcp_use_sketchfab
    del bpy.types.Scene.blendermcp_sketchfab_api_key
    del bpy.types.Scene.blendermcp_use_polypizza
    del bpy.types.Scene.blendermcp_polypizza_api_key
    del bpy.types.Scene.blendermcp_use_hunyuan3d
    del bpy.types.Scene.blendermcp_hunyuan3d_mode
    del bpy.types.Scene.blendermcp_hunyuan3d_secret_id
    del bpy.types.Scene.blendermcp_hunyuan3d_secret_key
    del bpy.types.Scene.blendermcp_hunyuan3d_api_url
    del bpy.types.Scene.blendermcp_hunyuan3d_octree_resolution
    del bpy.types.Scene.blendermcp_hunyuan3d_num_inference_steps
    del bpy.types.Scene.blendermcp_hunyuan3d_guidance_scale
    del bpy.types.Scene.blendermcp_hunyuan3d_texture

    print("BlenderMCP addon unregistered")

if __name__ == "__main__":
    register()
