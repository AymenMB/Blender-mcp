# Blender MCP Server v2.0.0 - Created by Aymen Mabrouk
# The ultimate Blender integration through the Model Context Protocol
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Optional
from threading import RLock
import os
from pathlib import Path
import base64
import urllib.request
import urllib.parse
from urllib.parse import urlparse
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlenderMCPServer")

# Default configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876

@dataclass
class BlenderConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict
    request_lock: Any = field(default_factory=RLock, repr=False)
    
    def connect(self) -> bool:
        """Connect to the Blender addon socket server"""
        if self.sock:
            return True
            
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(None)
            logger.info(f"Connected to Blender at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Blender: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Blender addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=8192):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a consistent timeout value that matches the addon's timeout
        sock.settimeout(180.0)  # Match the addon's timeout
        
        try:
            while True:
                try:
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        # If we get an empty chunk, the connection might be closed
                        if not chunks:  # If we haven't received anything yet, this is an error
                            raise Exception("Connection closed before receiving any data")
                        break
                    
                    chunks.append(chunk)
                    
                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        # If we get here, it parsed successfully
                        logger.info(f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    # If we hit a timeout during receiving, break the loop and try to use what we have
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {str(e)}")
                    raise  # Re-raise to be handled by the caller
        except socket.timeout:
            logger.warning("Socket timeout during chunked receive")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise
            
        # If we get here, we either timed out or broke out of the loop
        # Try to use what we have
        if chunks:
            data = b''.join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                # Try to parse what we have
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                # If we can't parse it, it's incomplete
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Blender and return the response"""
        with self.request_lock:
            if not self.sock and not self.connect():
                raise ConnectionError("Not connected to Blender")

            command = {
                "type": command_type,
                "params": params or {}
            }

            try:
                # Log the command being sent
                logger.info(f"Sending command: {command_type} with params: {params}")

                # Send the command
                self.sock.sendall(json.dumps(command).encode('utf-8'))
                logger.info(f"Command sent, waiting for response...")

                # Set a timeout for receiving - use the same timeout as in receive_full_response
                self.sock.settimeout(180.0)  # Match the addon's timeout

                # Receive the response using the improved receive_full_response method
                response_data = self.receive_full_response(self.sock)
                logger.info(f"Received {len(response_data)} bytes of data")

                response = json.loads(response_data.decode('utf-8'))
                logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")

                if response.get("status") == "error":
                    logger.error(f"Blender error: {response.get('message')}")
                    raise Exception(response.get("message", "Unknown error from Blender"))

                return response.get("result", {})
            except socket.timeout:
                logger.error("Socket timeout while waiting for response from Blender")
                # Don't try to reconnect here - let the get_blender_connection handle reconnection
                # Just invalidate the current socket so it will be recreated next time
                self.sock = None
                raise Exception("Timeout waiting for Blender response - try simplifying your request")
            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Socket connection error: {str(e)}")
                self.sock = None
                raise Exception(f"Connection to Blender lost: {str(e)}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from Blender: {str(e)}")
                # Try to log what was received
                if 'response_data' in locals() and response_data:
                    logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
                raise Exception(f"Invalid response from Blender: {str(e)}")
            except Exception as e:
                logger.error(f"Error communicating with Blender: {str(e)}")
                # Don't try to reconnect here - let the get_blender_connection handle reconnection
                self.sock = None
                raise Exception(f"Communication error with Blender: {str(e)}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    # We don't need to create a connection here since we're using the global connection
    # for resources and tools

    try:
        logger.info("BlenderMCP server v2.0.0 starting up")

        # Try to connect to Blender on startup to verify it's available
        try:
            blender = get_blender_connection()
            logger.info("Successfully connected to Blender on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Blender on startup: {str(e)}")
            logger.warning("Make sure the Blender addon is running before using Blender resources or tools")

        yield {}
    finally:
        global _blender_connection
        if _blender_connection:
            logger.info("Disconnecting from Blender on shutdown")
            _blender_connection.disconnect()
            _blender_connection = None
        logger.info("BlenderMCP server shut down")

# Create the MCP server with lifespan support
mcp = FastMCP(
    "BlenderMCP",
    lifespan=server_lifespan
)

# Resource endpoints

# Global connection for resources (since resources can't access context)
_blender_connection = None
_polyhaven_enabled = False  # Add this global variable

def get_blender_connection():
    """Get or create a persistent Blender connection"""
    global _blender_connection, _polyhaven_enabled  # Add _polyhaven_enabled to globals
    
    # If we have an existing connection, check if it's still valid
    if _blender_connection is not None:
        try:
            # First check if PolyHaven is enabled by sending a ping command
            result = _blender_connection.send_command("get_polyhaven_status")
            # Store the PolyHaven status globally
            _polyhaven_enabled = result.get("enabled", False)
            return _blender_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _blender_connection.disconnect()
            except:
                pass
            _blender_connection = None
    
    # Create a new connection if needed
    if _blender_connection is None:
        host = os.getenv("BLENDER_HOST", DEFAULT_HOST)
        port = int(os.getenv("BLENDER_PORT", DEFAULT_PORT))
        _blender_connection = BlenderConnection(host=host, port=port)
        if not _blender_connection.connect():
            logger.error("Failed to connect to Blender")
            _blender_connection = None
            raise Exception("Could not connect to Blender. Make sure the Blender addon is running.")
        logger.info("Created new persistent connection to Blender")
    
    return _blender_connection


def _handle_result(result: dict) -> dict:
    """Validate result from Blender addon - handlers return {'error': '...'} on failure.
    Since _execute_command_internal wraps these in {'status': 'success', 'result': ...},
    send_command strips the wrapper and returns just the result dict.
    We need to check if it contains an error key."""
    if isinstance(result, dict) and "error" in result:
        raise Exception(result["error"])
    return result


def _asset_manifest_path() -> Path:
    """Path for persistent local asset ingestion audit log."""
    base = Path(os.getenv("BLENDER_MCP_ASSET_CACHE", str(Path.home() / ".blender_mcp")))
    base.mkdir(parents=True, exist_ok=True)
    return base / "assets_manifest.json"


def _append_asset_manifest(entry: Dict[str, Any]) -> None:
    """Append one entry to local asset manifest JSON."""
    manifest_path = _asset_manifest_path()
    records: List[Dict[str, Any]] = []
    if manifest_path.exists():
        try:
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
        except Exception:
            records = []

    records.append(entry)
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _license_passes_filter(license_label: str, license_filter: str) -> bool:
    lbl = (license_label or "").strip().lower()
    lf = (license_filter or "all").strip().lower()
    if lf == "all":
        return True
    if lf == "cc0":
        return ("cc0" in lbl) or ("public domain" in lbl)
    if lf == "cc-by":
        return "cc-by" in lbl
    return True


def _normalize_polyhaven_assets(assets: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for asset_id, asset_data in (assets or {}).items():
        normalized.append({
            "source": "polyhaven",
            "asset_id": asset_id,
            "title": asset_data.get("name", asset_id),
            "category": category,
            "tags": asset_data.get("categories", []),
            "license_label": "CC0",
            "download_count": asset_data.get("download_count", 0),
            "source_url": f"https://polyhaven.com/a/{asset_id}",
        })
    return normalized


def _normalize_sketchfab_assets(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for model in models or []:
        license_data = model.get("license") or {}
        user_data = model.get("user") or {}
        thumbs = (model.get("thumbnails") or {}).get("images", [])
        thumb_url = thumbs[0].get("url") if thumbs else None
        uid = model.get("uid")
        normalized.append({
            "source": "sketchfab",
            "asset_id": uid,
            "title": model.get("name", uid),
            "category": "model",
            "tags": model.get("tags", []),
            "license_label": license_data.get("label", "Unknown"),
            "author": user_data.get("username", "Unknown"),
            "face_count": model.get("faceCount"),
            "downloadable": model.get("isDownloadable", False),
            "thumbnail_url": thumb_url,
            "source_url": f"https://sketchfab.com/3d-models/{uid}" if uid else None,
        })
    return normalized


def _github_api_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "blender-mcp"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _github_raw_url(repo: str, path: str, branch: str = "main") -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


@mcp.tool()
def search_asset_sources(
    ctx: Context,
    query: str,
    category: str = "model",
    max_results: int = 10,
    license_filter: str = "all",
    providers: Optional[List[str]] = None,
) -> str:
    """
    Federated search across enabled internet asset sources with normalized output.

    Parameters:
    - query: search text (e.g., "rusted metal barrel")
    - category: model | texture | hdri
    - max_results: max returned items
    - license_filter: all | cc0 | cc-by
    - providers: optional list from ["polyhaven", "sketchfab", "polypizza", "github_khronos"]
    """
    try:
        blender = get_blender_connection()
        category = (category or "model").lower()
        max_results = min(max(1, max_results), 50)

        requested = [p.lower() for p in (providers or ["polyhaven", "sketchfab", "polypizza", "github_khronos"])]
        all_candidates: List[Dict[str, Any]] = []
        used_providers: List[str] = []

        if "polyhaven" in requested:
            try:
                asset_type = {
                    "model": "models",
                    "texture": "textures",
                    "hdri": "hdris",
                }.get(category, "all")
                poly = blender.send_command("search_polyhaven_assets", {
                    "asset_type": asset_type,
                    "categories": query,
                })
                if isinstance(poly, dict) and "assets" in poly:
                    used_providers.append("polyhaven")
                    all_candidates.extend(_normalize_polyhaven_assets(poly.get("assets", {}), category))
            except Exception:
                pass

        if "sketchfab" in requested and category == "model":
            try:
                sk = blender.send_command("search_sketchfab_models", {
                    "query": query,
                    "count": max_results,
                    "downloadable": True,
                })
                if isinstance(sk, dict) and isinstance(sk.get("results"), list):
                    used_providers.append("sketchfab")
                    all_candidates.extend(_normalize_sketchfab_assets(sk.get("results", [])))
            except Exception:
                pass

        if "polypizza" in requested and category == "model":
            try:
                pz = blender.send_command("search_polypizza_models", {
                    "query": query,
                    "count": max_results,
                })
                if isinstance(pz, dict) and isinstance(pz.get("results"), list):
                    used_providers.append("polypizza")
                    for item in pz.get("results", []):
                        all_candidates.append({
                            "source": "polypizza",
                            "asset_id": item.get("id"),
                            "title": item.get("name", item.get("id")),
                            "category": "model",
                            "tags": [],
                            "license_label": item.get("license", "Unknown"),
                            "tri_count": item.get("triCount"),
                            "download_url": item.get("download"),
                            "thumbnail_url": item.get("thumbnail"),
                            "source_url": f"https://poly.pizza/m/{item.get('id')}" if item.get("id") else None,
                        })
            except Exception:
                pass

        if "github_khronos" in requested and category == "model":
            try:
                tree = _github_api_json("https://api.github.com/repos/KhronosGroup/glTF-Sample-Assets/git/trees/main?recursive=1")
                used_providers.append("github_khronos")
                q = (query or "").lower().strip()
                for node in tree.get("tree", []):
                    path = node.get("path", "")
                    if not path.lower().endswith((".glb", ".gltf")):
                        continue
                    if q and q not in path.lower():
                        continue
                    raw = _github_raw_url("KhronosGroup/glTF-Sample-Assets", path, "main")
                    all_candidates.append({
                        "source": "github_khronos",
                        "asset_id": raw,
                        "title": path.split("/")[-1],
                        "category": "model",
                        "tags": ["khronos", "gltf", "sample"],
                        "license_label": "Repository license (see source)",
                        "download_url": raw,
                        "source_url": f"https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/{path}",
                    })
            except Exception:
                pass

        filtered = [
            item for item in all_candidates
            if _license_passes_filter(item.get("license_label", ""), license_filter)
        ]

        if category == "model":
            filtered.sort(key=lambda x: (x.get("source") != "sketchfab", -(x.get("face_count") or 0)))
        else:
            filtered.sort(key=lambda x: (x.get("source") != "polyhaven", -(x.get("download_count") or 0)))

        result = {
            "query": query,
            "category": category,
            "license_filter": license_filter,
            "providers_used": used_providers,
            "count": min(len(filtered), max_results),
            "results": filtered[:max_results],
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in search_asset_sources: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def get_asset_license(ctx: Context, source: str, asset_id: str) -> str:
    """
    Retrieve normalized license metadata for an asset.

    Parameters:
    - source: polyhaven | sketchfab
    - asset_id: provider-specific asset id/uid
    """
    try:
        blender = get_blender_connection()
        src = (source or "").lower().strip()

        if src == "polyhaven":
            result = {
                "source": "polyhaven",
                "asset_id": asset_id,
                "license_code": "CC0-1.0",
                "license_label": "CC0",
                "attribution_required": False,
                "commercial_use_allowed": True,
                "source_url": f"https://polyhaven.com/a/{asset_id}",
            }
            return json.dumps(result, indent=2)

        if src == "sketchfab":
            result = _handle_result(blender.send_command("get_sketchfab_model_license", {"uid": asset_id}))
            return json.dumps(result, indent=2)

        if src == "polypizza":
            # For Poly Pizza, license is retrieved from search metadata in current flow.
            return json.dumps({
                "source": "polypizza",
                "asset_id": asset_id,
                "license_code": "Provider metadata required",
                "license_label": "See model metadata",
                "attribution_required": True,
                "commercial_use_allowed": True,
                "source_url": f"https://poly.pizza/m/{asset_id}",
            }, indent=2)

        if src == "github_khronos":
            return json.dumps({
                "source": "github_khronos",
                "asset_id": asset_id,
                "license_code": "Repository license",
                "license_label": "Check repository LICENSE",
                "attribution_required": True,
                "commercial_use_allowed": True,
                "source_url": "https://github.com/KhronosGroup/glTF-Sample-Assets",
            }, indent=2)

        return json.dumps({"error": f"Unsupported source: {source}"}, indent=2)
    except Exception as e:
        logger.error(f"Error in get_asset_license: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def download_asset(
    ctx: Context,
    source: str,
    asset_id: str,
    asset_type: str = "model",
    target_size: float = 1.0,
    format_preference: str = "glb",
    resolution: str = "1k",
) -> str:
    """
    Unified download/import entry point for supported sources.

    Parameters:
    - source: polyhaven | sketchfab
    - asset_id: provider asset id
    - asset_type: model | texture | hdri (used by polyhaven)
    - target_size: model normalization size in meters (used by sketchfab)
    - format_preference: preferred format hint
    - resolution: quality hint for polyhaven
    """
    try:
        blender = get_blender_connection()
        src = (source or "").lower().strip()
        atype = (asset_type or "model").lower().strip()

        if src == "sketchfab":
            result = _handle_result(blender.send_command("download_sketchfab_model", {
                "uid": asset_id,
                "normalize_size": True,
                "target_size": float(target_size),
            }))
        elif src == "polypizza":
            result = _handle_result(blender.send_command("download_polypizza_model", {
                "model_id": asset_id,
                "download_url": None,
                "target_size": float(target_size),
            }))
        elif src == "polyhaven":
            poly_type = {
                "model": "models",
                "texture": "textures",
                "hdri": "hdris",
            }.get(atype, "models")

            default_format = {
                "models": "gltf",
                "textures": "jpg",
                "hdris": "hdr",
            }[poly_type]
            file_format = (format_preference or default_format).lower()

            result = _handle_result(blender.send_command("download_polyhaven_asset", {
                "asset_id": asset_id,
                "asset_type": poly_type,
                "resolution": resolution,
                "file_format": file_format,
            }))
        elif src == "github_khronos":
            # asset_id is expected to be a raw downloadable URL from search_asset_sources
            parsed = urlparse(asset_id)
            if parsed.scheme not in ("http", "https"):
                return json.dumps({"error": "Invalid github_khronos asset_id URL"}, indent=2)
            suffix = ".glb" if asset_id.lower().endswith(".glb") else ".gltf"
            tmp_dir = tempfile.mkdtemp(prefix="blender_mcp_github_")
            local_path = os.path.join(tmp_dir, f"asset{suffix}")
            req = urllib.request.Request(asset_id, headers={"User-Agent": "blender-mcp"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp, open(local_path, "wb") as f:
                    f.write(resp.read())
                result = _handle_result(blender.send_command("import_file", {"filepath": local_path}))
            finally:
                # keep temp file for current Blender session reliability; cleanup best effort on process exit
                pass
        else:
            return json.dumps({"error": f"Unsupported source: {source}"}, indent=2)

        manifest_entry = {
            "asset_id": asset_id,
            "source": src,
            "asset_type": atype,
            "ingest_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "format_preference": format_preference,
            "resolution": resolution,
            "result": result,
        }
        try:
            _append_asset_manifest(manifest_entry)
        except Exception as log_error:
            logger.warning(f"Manifest append failed: {log_error}")

        return json.dumps({
            "success": True,
            "source": src,
            "asset_id": asset_id,
            "result": result,
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in download_asset: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def asset_provenance_report(ctx: Context, limit: int = 100) -> str:
    """
    Return a local audit log of ingested assets.

    Parameters:
    - limit: max entries returned from the most recent records
    """
    try:
        manifest_path = _asset_manifest_path()
        if not manifest_path.exists():
            return json.dumps({
                "count": 0,
                "manifest_path": str(manifest_path),
                "entries": [],
            }, indent=2)

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = []
        lim = min(max(1, int(limit)), 1000)
        entries = data[-lim:]
        return json.dumps({
            "count": len(entries),
            "manifest_path": str(manifest_path),
            "entries": entries,
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in asset_provenance_report: {str(e)}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def get_polypizza_status(ctx: Context) -> str:
    """
    Check if Poly Pizza integration is enabled in Blender.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_polypizza_status")
        return result.get("message", "Poly Pizza status unavailable")
    except Exception as e:
        logger.error(f"Error checking Poly Pizza status: {str(e)}")
        return f"Error checking Poly Pizza status: {str(e)}"


@mcp.tool()
def search_polypizza_models(ctx: Context, query: str, count: int = 20) -> str:
    """
    Search Poly Pizza low-poly models.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("search_polypizza_models", {
            "query": query,
            "count": max(1, min(count, 50)),
        })
        if "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error searching Poly Pizza models: {str(e)}")
        return f"Error searching Poly Pizza models: {str(e)}"


@mcp.tool()
def download_polypizza_model(ctx: Context, model_id: str = None, download_url: str = None, target_size: float = 1.0) -> str:
    """
    Download and import a Poly Pizza model.

    Parameters:
    - model_id: Poly Pizza model id (preferred)
    - download_url: direct download URL (optional override)
    - target_size: normalize largest dimension to this size in meters
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("download_polypizza_model", {
            "model_id": model_id,
            "download_url": download_url,
            "target_size": target_size,
        })
        if "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error downloading Poly Pizza model: {str(e)}")
        return f"Error downloading Poly Pizza model: {str(e)}"


@mcp.tool()
def search_github_khronos_models(ctx: Context, query: str = "", max_results: int = 20) -> str:
    """
    Search glTF sample models from Khronos GitHub repository.
    """
    try:
        tree = _github_api_json("https://api.github.com/repos/KhronosGroup/glTF-Sample-Assets/git/trees/main?recursive=1")
        q = (query or "").lower().strip()
        out: List[Dict[str, Any]] = []
        for node in tree.get("tree", []):
            path = node.get("path", "")
            if not path.lower().endswith((".glb", ".gltf")):
                continue
            if q and q not in path.lower():
                continue
            raw = _github_raw_url("KhronosGroup/glTF-Sample-Assets", path, "main")
            out.append({
                "id": raw,
                "name": path.split("/")[-1],
                "path": path,
                "download_url": raw,
                "source_url": f"https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/{path}",
            })
            if len(out) >= max(1, min(max_results, 100)):
                break
        return json.dumps({"count": len(out), "results": out}, indent=2)
    except Exception as e:
        logger.error(f"Error searching GitHub Khronos models: {str(e)}")
        return f"Error searching GitHub Khronos models: {str(e)}"


@mcp.tool()
def download_github_khronos_model(ctx: Context, raw_url: str) -> str:
    """
    Download and import a model from a raw.githubusercontent.com URL.
    """
    try:
        return download_asset(ctx, source="github_khronos", asset_id=raw_url, asset_type="model")
    except Exception as e:
        logger.error(f"Error downloading GitHub Khronos model: {str(e)}")
        return f"Error downloading GitHub Khronos model: {str(e)}"

@mcp.tool()
def get_scene_info(ctx: Context) -> str:
    """Get detailed information about the current Blender scene"""
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info")

        # Just return the JSON representation of what Blender sent us
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting scene info from Blender: {str(e)}")
        return f"Error getting scene info: {str(e)}"


@mcp.tool()
def ping_blender(ctx: Context) -> str:
    """
    Fast connectivity check.

    Use this first when starting a session to confirm Blender MCP is reachable.
    """
    try:
        blender = get_blender_connection()
        scene = blender.send_command("get_scene_info")
        return json.dumps({
            "connected": True,
            "scene_name": scene.get("name"),
            "object_count": scene.get("object_count"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2)
    except Exception as e:
        logger.error(f"Error pinging Blender: {str(e)}")
        return json.dumps({
            "connected": False,
            "error": str(e),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2)


@mcp.tool()
def get_mcp_capabilities(ctx: Context) -> str:
    """
    Return a concise capabilities snapshot for this MCP server.

    This helps an LLM quickly decide which workflow to use without guessing.
    """
    try:
        blender = get_blender_connection()

        # Core status (always expected to exist)
        scene = blender.send_command("get_scene_info")

        # Optional integrations (safe-by-default checks)
        integrations = {
            "polyhaven": {"enabled": False},
            "sketchfab": {"enabled": False},
            "polypizza": {"enabled": False},
            "github_khronos": {"enabled": True, "message": "Available (public GitHub source)"},
            "hyper3d": {"enabled": False},
            "hunyuan3d": {"enabled": False},
        }

        try:
            integrations["polyhaven"] = blender.send_command("get_polyhaven_status")
        except Exception:
            pass
        try:
            integrations["sketchfab"] = blender.send_command("get_sketchfab_status")
        except Exception:
            pass
        try:
            integrations["polypizza"] = blender.send_command("get_polypizza_status")
        except Exception:
            pass
        try:
            integrations["hyper3d"] = blender.send_command("get_hyper3d_status")
        except Exception:
            pass
        try:
            integrations["hunyuan3d"] = blender.send_command("get_hunyuan3d_status")
        except Exception:
            pass

        capabilities = {
            "connected": True,
            "scene": {
                "name": scene.get("name"),
                "object_count": scene.get("object_count"),
            },
            "core_tool_groups": [
                "scene_inspection",
                "object_creation_and_transform",
                "modifiers",
                "materials",
                "camera_and_lighting",
                "animation",
                "collections",
                "render",
                "export",
                "python_automation",
            ],
            "integrations": integrations,
            "notes": [
                "Dedicated tools are preferred over execute_blender_code for common tasks.",
                "search_blender_docs is optional and only needed for advanced API lookup.",
            ],
        }
        return json.dumps(capabilities, indent=2)
    except Exception as e:
        logger.error(f"Error getting MCP capabilities: {str(e)}")
        return json.dumps({
            "connected": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
def get_object_info(ctx: Context, object_name: str) -> str:
    """
    Get detailed information about a specific object in the Blender scene.
    
    Parameters:
    - object_name: The name of the object to get information about
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_info", {"name": object_name})
        
        # Just return the JSON representation of what Blender sent us
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting object info from Blender: {str(e)}")
        return f"Error getting object info: {str(e)}"

@mcp.tool()
def get_viewport_screenshot(ctx: Context, max_size: int = 800) -> Image:
    """
    Capture a screenshot of the current Blender 3D viewport.
    
    Parameters:
    - max_size: Maximum size in pixels for the largest dimension (default: 800)
    
    Returns the screenshot as an Image.
    """
    try:
        blender = get_blender_connection()
        
        # Create temp file path
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"blender_screenshot_{os.getpid()}.png")
        
        result = blender.send_command("get_viewport_screenshot", {
            "max_size": max_size,
            "filepath": temp_path,
            "format": "png"
        })
        
        if "error" in result:
            raise Exception(result["error"])
        
        if not os.path.exists(temp_path):
            raise Exception("Screenshot file was not created")
        
        # Read the file
        with open(temp_path, 'rb') as f:
            image_bytes = f.read()
        
        # Delete the temp file
        os.remove(temp_path)
        
        return Image(data=image_bytes, format="png")
        
    except Exception as e:
        logger.error(f"Error capturing screenshot: {str(e)}")
        raise Exception(f"Screenshot failed: {str(e)}")


@mcp.tool()
def execute_blender_code(ctx: Context, code: str) -> str:
    """
    Execute arbitrary Python code in Blender. Make sure to do it step-by-step by breaking it into smaller chunks.

    Parameters:
    - code: The Python code to execute
    """
    try:
        # Get the global connection
        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": code})
        return f"Code executed successfully: {result.get('result', '')}"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return f"Error executing code: {str(e)}"

@mcp.tool()
def get_polyhaven_categories(ctx: Context, asset_type: str = "hdris") -> str:
    """
    Get a list of categories for a specific asset type on Polyhaven.
    
    Parameters:
    - asset_type: The type of asset to get categories for (hdris, textures, models, all)
    """
    try:
        blender = get_blender_connection()
        if not _polyhaven_enabled:
            return "PolyHaven integration is disabled. Select it in the sidebar in BlenderMCP, then run it again."
        result = blender.send_command("get_polyhaven_categories", {"asset_type": asset_type})
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Format the categories in a more readable way
        categories = result["categories"]
        formatted_output = f"Categories for {asset_type}:\n\n"
        
        # Sort categories by count (descending)
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories:
            formatted_output += f"- {category}: {count} assets\n"
        
        return formatted_output
    except Exception as e:
        logger.error(f"Error getting Polyhaven categories: {str(e)}")
        return f"Error getting Polyhaven categories: {str(e)}"

@mcp.tool()
def search_polyhaven_assets(
    ctx: Context,
    asset_type: str = "all",
    categories: str = None
) -> str:
    """
    Search for assets on Polyhaven with optional filtering.
    
    Parameters:
    - asset_type: Type of assets to search for (hdris, textures, models, all)
    - categories: Optional comma-separated list of categories to filter by
    
    Returns a list of matching assets with basic information.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("search_polyhaven_assets", {
            "asset_type": asset_type,
            "categories": categories
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Format the assets in a more readable way
        assets = result["assets"]
        total_count = result["total_count"]
        returned_count = result["returned_count"]
        
        formatted_output = f"Found {total_count} assets"
        if categories:
            formatted_output += f" in categories: {categories}"
        formatted_output += f"\nShowing {returned_count} assets:\n\n"
        
        # Sort assets by download count (popularity)
        sorted_assets = sorted(assets.items(), key=lambda x: x[1].get("download_count", 0), reverse=True)
        
        for asset_id, asset_data in sorted_assets:
            formatted_output += f"- {asset_data.get('name', asset_id)} (ID: {asset_id})\n"
            formatted_output += f"  Type: {['HDRI', 'Texture', 'Model'][asset_data.get('type', 0)]}\n"
            formatted_output += f"  Categories: {', '.join(asset_data.get('categories', []))}\n"
            formatted_output += f"  Downloads: {asset_data.get('download_count', 'Unknown')}\n\n"
        
        return formatted_output
    except Exception as e:
        logger.error(f"Error searching Polyhaven assets: {str(e)}")
        return f"Error searching Polyhaven assets: {str(e)}"

@mcp.tool()
def download_polyhaven_asset(
    ctx: Context,
    asset_id: str,
    asset_type: str,
    resolution: str = "1k",
    file_format: str = None
) -> str:
    """
    Download and import a Polyhaven asset into Blender.
    
    Parameters:
    - asset_id: The ID of the asset to download
    - asset_type: The type of asset (hdris, textures, models)
    - resolution: The resolution to download (e.g., 1k, 2k, 4k)
    - file_format: Optional file format (e.g., hdr, exr for HDRIs; jpg, png for textures; gltf, fbx for models)
    
    Returns a message indicating success or failure.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("download_polyhaven_asset", {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "resolution": resolution,
            "file_format": file_format
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            message = result.get("message", "Asset downloaded and imported successfully")
            
            # Add additional information based on asset type
            if asset_type == "hdris":
                return f"{message}. The HDRI has been set as the world environment."
            elif asset_type == "textures":
                material_name = result.get("material", "")
                maps = ", ".join(result.get("maps", []))
                return f"{message}. Created material '{material_name}' with maps: {maps}."
            elif asset_type == "models":
                return f"{message}. The model has been imported into the current scene."
            else:
                return message
        else:
            return f"Failed to download asset: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error downloading Polyhaven asset: {str(e)}")
        return f"Error downloading Polyhaven asset: {str(e)}"

@mcp.tool()
def set_texture(
    ctx: Context,
    object_name: str,
    texture_id: str
) -> str:
    """
    Apply a previously downloaded Polyhaven texture to an object.
    
    Parameters:
    - object_name: Name of the object to apply the texture to
    - texture_id: ID of the Polyhaven texture to apply (must be downloaded first)
    
    Returns a message indicating success or failure.
    """
    try:
        # Get the global connection
        blender = get_blender_connection()
        result = blender.send_command("set_texture", {
            "object_name": object_name,
            "texture_id": texture_id
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            material_name = result.get("material", "")
            maps = ", ".join(result.get("maps", []))
            
            # Add detailed material info
            material_info = result.get("material_info", {})
            node_count = material_info.get("node_count", 0)
            has_nodes = material_info.get("has_nodes", False)
            texture_nodes = material_info.get("texture_nodes", [])
            
            output = f"Successfully applied texture '{texture_id}' to {object_name}.\n"
            output += f"Using material '{material_name}' with maps: {maps}.\n\n"
            output += f"Material has nodes: {has_nodes}\n"
            output += f"Total node count: {node_count}\n\n"
            
            if texture_nodes:
                output += "Texture nodes:\n"
                for node in texture_nodes:
                    output += f"- {node['name']} using image: {node['image']}\n"
                    if node['connections']:
                        output += "  Connections:\n"
                        for conn in node['connections']:
                            output += f"    {conn}\n"
            else:
                output += "No texture nodes found in the material.\n"
            
            return output
        else:
            return f"Failed to apply texture: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error applying texture: {str(e)}")
        return f"Error applying texture: {str(e)}"

@mcp.tool()
def get_polyhaven_status(ctx: Context) -> str:
    """
    Check if PolyHaven integration is enabled in Blender.
    Returns a message indicating whether PolyHaven features are available.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_polyhaven_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += "PolyHaven is good at Textures, and has a wider variety of textures than Sketchfab."
        return message
    except Exception as e:
        logger.error(f"Error checking PolyHaven status: {str(e)}")
        return f"Error checking PolyHaven status: {str(e)}"

@mcp.tool()
def get_hyper3d_status(ctx: Context) -> str:
    """
    Check if Hyper3D Rodin integration is enabled in Blender.
    Returns a message indicating whether Hyper3D Rodin features are available.

    Don't emphasize the key type in the returned message, but sliently remember it. 
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_hyper3d_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += ""
        return message
    except Exception as e:
        logger.error(f"Error checking Hyper3D status: {str(e)}")
        return f"Error checking Hyper3D status: {str(e)}"

@mcp.tool()
def get_sketchfab_status(ctx: Context) -> str:
    """
    Check if Sketchfab integration is enabled in Blender.
    Returns a message indicating whether Sketchfab features are available.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_sketchfab_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += "Sketchfab is good at Realistic models, and has a wider variety of models than PolyHaven."        
        return message
    except Exception as e:
        logger.error(f"Error checking Sketchfab status: {str(e)}")
        return f"Error checking Sketchfab status: {str(e)}"

@mcp.tool()
def search_sketchfab_models(
    ctx: Context,
    query: str,
    categories: str = None,
    count: int = 20,
    downloadable: bool = True
) -> str:
    """
    Search for models on Sketchfab with optional filtering.

    Parameters:
    - query: Text to search for
    - categories: Optional comma-separated list of categories
    - count: Maximum number of results to return (default 20)
    - downloadable: Whether to include only downloadable models (default True)

    Returns a formatted list of matching models.
    """
    try:
        blender = get_blender_connection()
        logger.info(f"Searching Sketchfab models with query: {query}, categories: {categories}, count: {count}, downloadable: {downloadable}")
        result = blender.send_command("search_sketchfab_models", {
            "query": query,
            "categories": categories,
            "count": count,
            "downloadable": downloadable
        })
        
        if "error" in result:
            logger.error(f"Error from Sketchfab search: {result['error']}")
            return f"Error: {result['error']}"
        
        # Safely get results with fallbacks for None
        if result is None:
            logger.error("Received None result from Sketchfab search")
            return "Error: Received no response from Sketchfab search"
            
        # Format the results
        models = result.get("results", []) or []
        if not models:
            return f"No models found matching '{query}'"
            
        formatted_output = f"Found {len(models)} models matching '{query}':\n\n"
        
        for model in models:
            if model is None:
                continue
                
            model_name = model.get("name", "Unnamed model")
            model_uid = model.get("uid", "Unknown ID")
            formatted_output += f"- {model_name} (UID: {model_uid})\n"
            
            # Get user info with safety checks
            user = model.get("user") or {}
            username = user.get("username", "Unknown author") if isinstance(user, dict) else "Unknown author"
            formatted_output += f"  Author: {username}\n"
            
            # Get license info with safety checks
            license_data = model.get("license") or {}
            license_label = license_data.get("label", "Unknown") if isinstance(license_data, dict) else "Unknown"
            formatted_output += f"  License: {license_label}\n"
            
            # Add face count and downloadable status
            face_count = model.get("faceCount", "Unknown")
            is_downloadable = "Yes" if model.get("isDownloadable") else "No"
            formatted_output += f"  Face count: {face_count}\n"
            formatted_output += f"  Downloadable: {is_downloadable}\n\n"
        
        return formatted_output
    except Exception as e:
        logger.error(f"Error searching Sketchfab models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error searching Sketchfab models: {str(e)}"

@mcp.tool()
def get_sketchfab_model_preview(
    ctx: Context,
    uid: str
) -> Image:
    """
    Get a preview thumbnail of a Sketchfab model by its UID.
    Use this to visually confirm a model before downloading.
    
    Parameters:
    - uid: The unique identifier of the Sketchfab model (obtained from search_sketchfab_models)
    
    Returns the model's thumbnail as an Image for visual confirmation.
    """
    try:
        blender = get_blender_connection()
        logger.info(f"Getting Sketchfab model preview for UID: {uid}")
        
        result = blender.send_command("get_sketchfab_model_preview", {"uid": uid})
        
        if result is None:
            raise Exception("Received no response from Blender")
        
        if "error" in result:
            raise Exception(result["error"])
        
        # Decode base64 image data
        image_data = base64.b64decode(result["image_data"])
        img_format = result.get("format", "jpeg")
        
        # Log model info
        model_name = result.get("model_name", "Unknown")
        author = result.get("author", "Unknown")
        logger.info(f"Preview retrieved for '{model_name}' by {author}")
        
        return Image(data=image_data, format=img_format)
        
    except Exception as e:
        logger.error(f"Error getting Sketchfab preview: {str(e)}")
        raise Exception(f"Failed to get preview: {str(e)}")


@mcp.tool()
def download_sketchfab_model(
    ctx: Context,
    uid: str,
    target_size: float
) -> str:
    """
    Download and import a Sketchfab model by its UID.
    The model will be scaled so its largest dimension equals target_size.
    
    Parameters:
    - uid: The unique identifier of the Sketchfab model
    - target_size: REQUIRED. The target size in Blender units/meters for the largest dimension.
                  You must specify the desired size for the model.
                  Examples:
                  - Chair: target_size=1.0 (1 meter tall)
                  - Table: target_size=0.75 (75cm tall)
                  - Car: target_size=4.5 (4.5 meters long)
                  - Person: target_size=1.7 (1.7 meters tall)
                  - Small object (cup, phone): target_size=0.1 to 0.3
    
    Returns a message with import details including object names, dimensions, and bounding box.
    The model must be downloadable and you must have proper access rights.
    """
    try:
        blender = get_blender_connection()
        logger.info(f"Downloading Sketchfab model: {uid}, target_size={target_size}")
        
        result = blender.send_command("download_sketchfab_model", {
            "uid": uid,
            "normalize_size": True,  # Always normalize
            "target_size": target_size
        })
        
        if result is None:
            logger.error("Received None result from Sketchfab download")
            return "Error: Received no response from Sketchfab download request"
            
        if "error" in result:
            logger.error(f"Error from Sketchfab download: {result['error']}")
            return f"Error: {result['error']}"
        
        if result.get("success"):
            imported_objects = result.get("imported_objects", [])
            object_names = ", ".join(imported_objects) if imported_objects else "none"
            
            output = f"Successfully imported model.\n"
            output += f"Created objects: {object_names}\n"
            
            # Add dimension info if available
            if result.get("dimensions"):
                dims = result["dimensions"]
                output += f"Dimensions (X, Y, Z): {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} meters\n"
            
            # Add bounding box info if available
            if result.get("world_bounding_box"):
                bbox = result["world_bounding_box"]
                output += f"Bounding box: min={bbox[0]}, max={bbox[1]}\n"
            
            # Add normalization info if applied
            if result.get("normalized"):
                scale = result.get("scale_applied", 1.0)
                output += f"Size normalized: scale factor {scale:.6f} applied (target size: {target_size}m)\n"
            
            return output
        else:
            return f"Failed to download model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error downloading Sketchfab model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error downloading Sketchfab model: {str(e)}"

def _process_bbox(original_bbox: list[float] | list[int] | None) -> list[int] | None:
    if original_bbox is None:
        return None
    if all(isinstance(i, int) for i in original_bbox):
        return original_bbox
    if any(i<=0 for i in original_bbox):
        raise ValueError("Incorrect number range: bbox must be bigger than zero!")
    return [int(float(i) / max(original_bbox) * 100) for i in original_bbox] if original_bbox else None

@mcp.tool()
def generate_hyper3d_model_via_text(
    ctx: Context,
    text_prompt: str,
    bbox_condition: list[float]=None
) -> str:
    """
    Generate 3D asset using Hyper3D by giving description of the desired asset, and import the asset into Blender.
    The 3D asset has built-in materials.
    The generated model has a normalized size, so re-scaling after generation can be useful.

    Parameters:
    - text_prompt: A short description of the desired model in **English**.
    - bbox_condition: Optional. If given, it has to be a list of floats of length 3. Controls the ratio between [Length, Width, Height] of the model.

    Returns a message indicating success or failure.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_rodin_job", {
            "text_prompt": text_prompt,
            "images": None,
            "bbox_condition": _process_bbox(bbox_condition),
        })
        succeed = result.get("submit_time", False)
        if succeed:
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        else:
            return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def generate_hyper3d_model_via_images(
    ctx: Context,
    input_image_paths: list[str]=None,
    input_image_urls: list[str]=None,
    bbox_condition: list[float]=None
) -> str:
    """
    Generate 3D asset using Hyper3D by giving images of the wanted asset, and import the generated asset into Blender.
    The 3D asset has built-in materials.
    The generated model has a normalized size, so re-scaling after generation can be useful.
    
    Parameters:
    - input_image_paths: The **absolute** paths of input images. Even if only one image is provided, wrap it into a list. Required if Hyper3D Rodin in MAIN_SITE mode.
    - input_image_urls: The URLs of input images. Even if only one image is provided, wrap it into a list. Required if Hyper3D Rodin in FAL_AI mode.
    - bbox_condition: Optional. If given, it has to be a list of ints of length 3. Controls the ratio between [Length, Width, Height] of the model.

    Only one of {input_image_paths, input_image_urls} should be given at a time, depending on the Hyper3D Rodin's current mode.
    Returns a message indicating success or failure.
    """
    if input_image_paths is not None and input_image_urls is not None:
        return f"Error: Conflict parameters given!"
    if input_image_paths is None and input_image_urls is None:
        return f"Error: No image given!"
    if input_image_paths is not None:
        if not all(os.path.exists(i) for i in input_image_paths):
            return "Error: not all image paths are valid!"
        images = []
        for path in input_image_paths:
            with open(path, "rb") as f:
                images.append(
                    (Path(path).suffix, base64.b64encode(f.read()).decode("ascii"))
                )
    elif input_image_urls is not None:
        parsed_urls = [urlparse(i) for i in input_image_urls]
        if not all(p.scheme in ("http", "https") and p.netloc for p in parsed_urls):
            return "Error: not all image URLs are valid!"
        images = input_image_urls.copy()
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_rodin_job", {
            "text_prompt": None,
            "images": images,
            "bbox_condition": _process_bbox(bbox_condition),
        })
        succeed = result.get("submit_time", False)
        if succeed:
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        else:
            return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def poll_rodin_job_status(
    ctx: Context,
    subscription_key: str=None,
    request_id: str=None,
):
    """
    Check if the Hyper3D Rodin generation task is completed.

    For Hyper3D Rodin mode MAIN_SITE:
        Parameters:
        - subscription_key: The subscription_key given in the generate model step.

        Returns a list of status. The task is done if all status are "Done".
        If "Failed" showed up, the generating process failed.
        This is a polling API, so only proceed if the status are finally determined ("Done" or "Canceled").

    For Hyper3D Rodin mode FAL_AI:
        Parameters:
        - request_id: The request_id given in the generate model step.

        Returns the generation task status. The task is done if status is "COMPLETED".
        The task is in progress if status is "IN_PROGRESS".
        If status other than "COMPLETED", "IN_PROGRESS", "IN_QUEUE" showed up, the generating process might be failed.
        This is a polling API, so only proceed if the status are finally determined ("COMPLETED" or some failed state).
    """
    try:
        blender = get_blender_connection()
        kwargs = {}
        if subscription_key:
            kwargs = {
                "subscription_key": subscription_key,
            }
        elif request_id:
            kwargs = {
                "request_id": request_id,
            }
        result = blender.send_command("poll_rodin_job_status", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def import_generated_asset(
    ctx: Context,
    name: str,
    task_uuid: str=None,
    request_id: str=None,
):
    """
    Import the asset generated by Hyper3D Rodin after the generation task is completed.

    Parameters:
    - name: The name of the object in scene
    - task_uuid: For Hyper3D Rodin mode MAIN_SITE: The task_uuid given in the generate model step.
    - request_id: For Hyper3D Rodin mode FAL_AI: The request_id given in the generate model step.

    Only give one of {task_uuid, request_id} based on the Hyper3D Rodin Mode!
    Return if the asset has been imported successfully.
    """
    try:
        blender = get_blender_connection()
        kwargs = {
            "name": name
        }
        if task_uuid:
            kwargs["task_uuid"] = task_uuid
        elif request_id:
            kwargs["request_id"] = request_id
        result = blender.send_command("import_generated_asset", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def get_hunyuan3d_status(ctx: Context) -> str:
    """
    Check if Hunyuan3D integration is enabled in Blender.
    Returns a message indicating whether Hunyuan3D features are available.

    Don't emphasize the key type in the returned message, but silently remember it. 
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_hunyuan3d_status")
        message = result.get("message", "")
        return message
    except Exception as e:
        logger.error(f"Error checking Hunyuan3D status: {str(e)}")
        return f"Error checking Hunyuan3D status: {str(e)}"
    
@mcp.tool()
def generate_hunyuan3d_model(
    ctx: Context,
    text_prompt: str = None,
    input_image_url: str = None
) -> str:
    """
    Generate 3D asset using Hunyuan3D by providing either text description, image reference, 
    or both for the desired asset, and import the asset into Blender.
    The 3D asset has built-in materials.
    
    Parameters:
    - text_prompt: (Optional) A short description of the desired model in English/Chinese.
    - input_image_url: (Optional) The local or remote url of the input image. Accepts None if only using text prompt.

    Returns: 
    - When successful, returns a JSON with job_id (format: "job_xxx") indicating the task is in progress
    - When the job completes, the status will change to "DONE" indicating the model has been imported
    - Returns error message if the operation fails
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_hunyuan_job", {
            "text_prompt": text_prompt,
            "image": input_image_url,
        })
        if "JobId" in result.get("Response", {}):
            job_id = result["Response"]["JobId"]
            formatted_job_id = f"job_{job_id}"
            return json.dumps({
                "job_id": formatted_job_id,
            })
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hunyuan3D task: {str(e)}")
        return f"Error generating Hunyuan3D task: {str(e)}"
    
@mcp.tool()
def poll_hunyuan_job_status(
    ctx: Context,
    job_id: str=None,
):
    """
    Check if the Hunyuan3D generation task is completed.

    For Hunyuan3D:
        Parameters:
        - job_id: The job_id given in the generate model step.

        Returns the generation task status. The task is done if status is "DONE".
        The task is in progress if status is "RUN".
        If status is "DONE", returns ResultFile3Ds, which is the generated ZIP model path
        When the status is "DONE", the response includes a field named ResultFile3Ds that contains the generated ZIP file path of the 3D model in OBJ format.
        This is a polling API, so only proceed if the status are finally determined ("DONE" or some failed state).
    """
    try:
        blender = get_blender_connection()
        kwargs = {
            "job_id": job_id,
        }
        result = blender.send_command("poll_hunyuan_job_status", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hunyuan3D task: {str(e)}")
        return f"Error generating Hunyuan3D task: {str(e)}"

@mcp.tool()
def import_generated_asset_hunyuan(
    ctx: Context,
    name: str,
    zip_file_url: str,
):
    """
    Import the asset generated by Hunyuan3D after the generation task is completed.

    Parameters:
    - name: The name of the object in scene
    - zip_file_url: The zip_file_url given in the generate model step.

    Return if the asset has been imported successfully.
    """
    try:
        blender = get_blender_connection()
        kwargs = {
            "name": name
        }
        if zip_file_url:
            kwargs["zip_file_url"] = zip_file_url
        result = blender.send_command("import_generated_asset_hunyuan", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hunyuan3D task: {str(e)}")
        return f"Error generating Hunyuan3D task: {str(e)}"


# ============================================================================
# NEW DEDICATED BLENDER TOOLS - Direct manipulation without execute_blender_code
# ============================================================================

@mcp.tool()
def create_object(
    ctx: Context,
    type: str,
    name: str = None,
    location: list = None,
    rotation: list = None,
    scale: list = None
) -> str:
    """
    Create a new object in the Blender scene.

    Parameters:
    - type: Object type to create. Options: CUBE, SPHERE, CYLINDER, CONE, TORUS, PLANE,
            CIRCLE, GRID, MONKEY, UV_SPHERE, ICO_SPHERE, EMPTY, CAMERA, LIGHT_POINT,
            LIGHT_SUN, LIGHT_SPOT, LIGHT_AREA, BEZIER_CURVE, NURBS_CURVE, TEXT
    - name: Optional name for the object
    - location: Optional [x, y, z] position (default [0, 0, 0])
    - rotation: Optional [x, y, z] rotation in degrees
    - scale: Optional [x, y, z] scale
    """
    try:
        blender = get_blender_connection()
        params = {"type": type.upper()}
        if name: params["name"] = name
        if location: params["location"] = location
        if rotation: params["rotation"] = rotation
        if scale: params["scale"] = scale
        result = _handle_result(blender.send_command("create_object", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error creating object: {str(e)}")
        return f"Error creating object: {str(e)}"


@mcp.tool()
def delete_object(ctx: Context, name: str, delete_children: bool = False) -> str:
    """
    Delete an object from the Blender scene.

    Parameters:
    - name: Name of the object to delete
    - delete_children: If True, also delete child objects
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("delete_object", {
            "name": name,
            "delete_children": delete_children
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error deleting object: {str(e)}")
        return f"Error deleting object: {str(e)}"


@mcp.tool()
def set_transform(
    ctx: Context,
    object_name: str,
    location: list = None,
    rotation: list = None,
    scale: list = None
) -> str:
    """
    Set the transform (location, rotation, scale) of an object.

    Parameters:
    - object_name: Name of the object to transform
    - location: Optional [x, y, z] world position
    - rotation: Optional [x, y, z] rotation in degrees
    - scale: Optional [x, y, z] scale factors
    """
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name}
        if location is not None: params["location"] = location
        if rotation is not None: params["rotation"] = rotation
        if scale is not None: params["scale"] = scale
        result = _handle_result(blender.send_command("set_transform", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error setting transform: {str(e)}")
        return f"Error setting transform: {str(e)}"


@mcp.tool()
def apply_transforms(
    ctx: Context,
    object_name: str,
    location: bool = True,
    rotation: bool = True,
    scale: bool = True
) -> str:
    """
    Apply (freeze) transforms on an object, making current transforms the new basis.

    Parameters:
    - object_name: Name of the object
    - location: Apply location transform (default True)
    - rotation: Apply rotation transform (default True)
    - scale: Apply scale transform (default True)
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("apply_transforms", {
            "object_name": object_name,
            "location": location,
            "rotation": rotation,
            "scale": scale
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error applying transforms: {str(e)}")
        return f"Error applying transforms: {str(e)}"


@mcp.tool()
def add_modifier(
    ctx: Context,
    object_name: str,
    modifier_type: str,
    modifier_name: str = None,
    params: dict = None
) -> str:
    """
    Add a modifier to an object.

    Parameters:
    - object_name: Name of the object
    - modifier_type: Type of modifier (e.g. SUBSURF, MIRROR, BOOLEAN, ARRAY, BEVEL,
                     SOLIDIFY, DECIMATE, SMOOTH, WIREFRAME, REMESH, SKIN, SHRINKWRAP,
                     SIMPLE_DEFORM, CURVE, ARMATURE, LATTICE, DISPLACE, WAVE, etc.)
    - modifier_name: Optional custom name for the modifier
    - params: Optional dict of modifier-specific parameters (e.g. {"levels": 3} for SUBSURF)
    """
    try:
        blender = get_blender_connection()
        cmd_params = {
            "object_name": object_name,
            "modifier_type": modifier_type.upper()
        }
        if modifier_name: cmd_params["modifier_name"] = modifier_name
        if params: cmd_params["params"] = params
        result = _handle_result(blender.send_command("add_modifier", cmd_params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error adding modifier: {str(e)}")
        return f"Error adding modifier: {str(e)}"


@mcp.tool()
def remove_modifier(ctx: Context, object_name: str, modifier_name: str) -> str:
    """
    Remove a modifier from an object.

    Parameters:
    - object_name: Name of the object
    - modifier_name: Name of the modifier to remove
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("remove_modifier", {
            "object_name": object_name,
            "modifier_name": modifier_name
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error removing modifier: {str(e)}")
        return f"Error removing modifier: {str(e)}"


@mcp.tool()
def apply_modifier(ctx: Context, object_name: str, modifier_name: str) -> str:
    """
    Apply a modifier on an object, making its effect permanent.

    Parameters:
    - object_name: Name of the object
    - modifier_name: Name of the modifier to apply
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("apply_modifier", {
            "object_name": object_name,
            "modifier_name": modifier_name
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error applying modifier: {str(e)}")
        return f"Error applying modifier: {str(e)}"


@mcp.tool()
def create_material(
    ctx: Context,
    name: str,
    base_color: list = None,
    metallic: float = 0.0,
    roughness: float = 0.5,
    emission_color: list = None,
    emission_strength: float = 0.0,
    alpha: float = 1.0,
    use_nodes: bool = True
) -> str:
    """
    Create a new material with Principled BSDF properties.

    Parameters:
    - name: Name of the material
    - base_color: [R, G, B, A] color values (0.0-1.0), e.g. [1.0, 0.0, 0.0, 1.0] for red
    - metallic: Metallic factor 0.0-1.0
    - roughness: Roughness factor 0.0-1.0
    - emission_color: [R, G, B, A] emission color
    - emission_strength: Emission strength (0.0 = no emission)
    - alpha: Opacity 0.0-1.0
    - use_nodes: Whether to use shader nodes (default True)
    """
    try:
        blender = get_blender_connection()
        params = {
            "name": name,
            "metallic": metallic,
            "roughness": roughness,
            "emission_strength": emission_strength,
            "alpha": alpha,
            "use_nodes": use_nodes
        }
        if base_color: params["base_color"] = base_color
        if emission_color: params["emission_color"] = emission_color
        result = _handle_result(blender.send_command("create_material", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error creating material: {str(e)}")
        return f"Error creating material: {str(e)}"


@mcp.tool()
def assign_material(ctx: Context, object_name: str, material_name: str) -> str:
    """
    Assign an existing material to an object.

    Parameters:
    - object_name: Name of the object
    - material_name: Name of the material to assign
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("assign_material", {
            "object_name": object_name,
            "material_name": material_name
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error assigning material: {str(e)}")
        return f"Error assigning material: {str(e)}"


@mcp.tool()
def render_image(
    ctx: Context,
    filepath: str = None,
    resolution_x: int = 1920,
    resolution_y: int = 1080,
    engine: str = None,
    samples: int = None
) -> str:
    """
    Render the current scene to an image file.

    Parameters:
    - filepath: Output file path (default: //render.png in blend file directory)
    - resolution_x: Horizontal resolution in pixels
    - resolution_y: Vertical resolution in pixels
    - engine: Render engine: BLENDER_EEVEE, CYCLES, or BLENDER_WORKBENCH
    - samples: Number of render samples (affects quality/speed)
    """
    try:
        blender = get_blender_connection()
        params = {
            "resolution_x": resolution_x,
            "resolution_y": resolution_y
        }
        if filepath: params["filepath"] = filepath
        if engine: params["engine"] = engine.upper()
        if samples: params["samples"] = samples
        result = _handle_result(blender.send_command("render_image", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error rendering image: {str(e)}")
        return f"Error rendering image: {str(e)}"


@mcp.tool()
def set_camera(
    ctx: Context,
    name: str = None,
    location: list = None,
    rotation: list = None,
    focal_length: float = None,
    look_at: list = None,
    set_active: bool = True
) -> str:
    """
    Configure a camera in the scene. Creates the camera if it doesn't exist.

    Parameters:
    - name: Camera name (default: "Camera")
    - location: [x, y, z] position
    - rotation: [x, y, z] rotation in degrees
    - focal_length: Lens focal length in mm
    - look_at: [x, y, z] point for the camera to look at (overrides rotation)
    - set_active: Set as the active scene camera (default True)
    """
    try:
        blender = get_blender_connection()
        params = {"set_active": set_active}
        if name: params["name"] = name
        if location: params["location"] = location
        if rotation: params["rotation"] = rotation
        if focal_length: params["focal_length"] = focal_length
        if look_at: params["look_at"] = look_at
        result = _handle_result(blender.send_command("set_camera", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error setting camera: {str(e)}")
        return f"Error setting camera: {str(e)}"


@mcp.tool()
def add_light(
    ctx: Context,
    type: str = "POINT",
    name: str = None,
    location: list = None,
    rotation: list = None,
    energy: float = 1000.0,
    color: list = None,
    size: float = None
) -> str:
    """
    Add a light to the scene.

    Parameters:
    - type: Light type: POINT, SUN, SPOT, AREA
    - name: Optional name for the light
    - location: [x, y, z] position
    - rotation: [x, y, z] rotation in degrees (important for SUN, SPOT, AREA)
    - energy: Light power in watts (default 1000)
    - color: [R, G, B] light color (0.0-1.0)
    - size: Light size/radius for soft shadows
    """
    try:
        blender = get_blender_connection()
        params = {
            "type": type.upper(),
            "energy": energy
        }
        if name: params["name"] = name
        if location: params["location"] = location
        if rotation: params["rotation"] = rotation
        if color: params["color"] = color
        if size is not None: params["size"] = size
        result = _handle_result(blender.send_command("add_light", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error adding light: {str(e)}")
        return f"Error adding light: {str(e)}"


@mcp.tool()
def manage_collection(
    ctx: Context,
    action: str,
    name: str = None,
    object_name: str = None,
    parent_name: str = None
) -> str:
    """
    Manage Blender collections (groups of objects).

    Parameters:
    - action: Action to perform: "create", "list", "move", "delete"
    - name: Collection name (for create/delete/move target)
    - object_name: Object to move (for "move" action)
    - parent_name: Parent collection name (for "create" action, default: Scene Collection)
    """
    try:
        blender = get_blender_connection()
        params = {"action": action.lower()}
        if name: params["name"] = name
        if object_name: params["object_name"] = object_name
        if parent_name: params["parent_name"] = parent_name
        result = _handle_result(blender.send_command("manage_collection", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error managing collection: {str(e)}")
        return f"Error managing collection: {str(e)}"


@mcp.tool()
def set_keyframe(
    ctx: Context,
    object_name: str,
    frame: int,
    data_path: str = None,
    location: list = None,
    rotation: list = None,
    scale: list = None,
    value: float = None
) -> str:
    """
    Insert a keyframe for animation on an object.

    Parameters:
    - object_name: Name of the object to animate
    - frame: Frame number to insert the keyframe at
    - data_path: Blender property path (e.g. "location", "rotation_euler", "scale",
                 "hide_viewport", or custom like "modifiers[\"Subsurf\"].levels")
    - location: Optional [x, y, z] - set location and key it
    - rotation: Optional [x, y, z] in degrees - set rotation and key it
    - scale: Optional [x, y, z] - set scale and key it
    - value: Optional value for custom data_path keyframes
    """
    try:
        blender = get_blender_connection()
        params = {
            "object_name": object_name,
            "frame": frame
        }
        if data_path: params["data_path"] = data_path
        if location: params["location"] = location
        if rotation: params["rotation"] = rotation
        if scale: params["scale"] = scale
        if value is not None: params["value"] = value
        result = _handle_result(blender.send_command("set_keyframe", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error setting keyframe: {str(e)}")
        return f"Error setting keyframe: {str(e)}"


@mcp.tool()
def export_scene(
    ctx: Context,
    filepath: str,
    format: str = "GLTF",
    selected_only: bool = False
) -> str:
    """
    Export the scene or selected objects to a file.

    Parameters:
    - filepath: Output file path (extension will be added if missing)
    - format: Export format: GLTF, GLB, FBX, OBJ, STL, PLY, ABC (Alembic), USD
    - selected_only: Only export selected objects (default False)
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("export_scene", {
            "filepath": filepath,
            "format": format.upper(),
            "selected_only": selected_only
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error exporting scene: {str(e)}")
        return f"Error exporting scene: {str(e)}"


@mcp.tool()
def duplicate_object(
    ctx: Context,
    name: str,
    new_name: str = None,
    linked: bool = False
) -> str:
    """
    Duplicate an object in the scene.

    Parameters:
    - name: Name of the object to duplicate
    - new_name: Optional name for the duplicate
    - linked: If True, create a linked duplicate (shares mesh data)
    """
    try:
        blender = get_blender_connection()
        params = {"name": name, "linked": linked}
        if new_name: params["new_name"] = new_name
        result = _handle_result(blender.send_command("duplicate_object", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error duplicating object: {str(e)}")
        return f"Error duplicating object: {str(e)}"


@mcp.tool()
def join_objects(
    ctx: Context,
    target_name: str,
    source_names: list
) -> str:
    """
    Join multiple objects into a single object. All source objects are merged into target.

    Parameters:
    - target_name: Name of the target object (will receive all geometry)
    - source_names: List of object names to merge into the target
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("join_objects", {
            "target_name": target_name,
            "source_names": source_names
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error joining objects: {str(e)}")
        return f"Error joining objects: {str(e)}"


@mcp.tool()
def set_parent(
    ctx: Context,
    child_name: str,
    parent_name: str = None,
    keep_transform: bool = True
) -> str:
    """
    Set or clear parent-child relationship between objects.

    Parameters:
    - child_name: Name of the child object
    - parent_name: Name of the parent object. If None, clears the parent.
    - keep_transform: Keep the child's world transform when parenting (default True)
    """
    try:
        blender = get_blender_connection()
        params = {"child_name": child_name, "keep_transform": keep_transform}
        if parent_name: params["parent_name"] = parent_name
        result = _handle_result(blender.send_command("set_parent", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error setting parent: {str(e)}")
        return f"Error setting parent: {str(e)}"


@mcp.tool()
def select_objects(
    ctx: Context,
    object_names: list[str] = None,
    clear: bool = True,
    set_active: str = None
) -> str:
    """
    Select objects by name and optionally set the active object.

    Parameters:
    - object_names: List of object names to select
    - clear: Deselect all first (default True)
    - set_active: Optional object name to set as active
    """
    try:
        blender = get_blender_connection()
        params = {"clear": clear}
        if object_names is not None:
            params["object_names"] = object_names
        if set_active:
            params["set_active"] = set_active
        result = _handle_result(blender.send_command("select_objects", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error selecting objects: {str(e)}")
        return f"Error selecting objects: {str(e)}"


@mcp.tool()
def frame_control(
    ctx: Context,
    action: str = "get",
    frame: int = None,
    start: int = None,
    end: int = None,
    fps: int = None
) -> str:
    """
    Control timeline frame settings.

    Parameters:
    - action: get | set | range | fps
    - frame: Required for action='set'
    - start, end: Optional for action='range'
    - fps: Required for action='fps'
    """
    try:
        blender = get_blender_connection()
        params = {"action": action}
        if frame is not None:
            params["frame"] = frame
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        if fps is not None:
            params["fps"] = fps
        result = _handle_result(blender.send_command("frame_control", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error controlling frame: {str(e)}")
        return f"Error controlling frame: {str(e)}"


@mcp.tool()
def save_blend_file(
    ctx: Context,
    filepath: str = None,
    pack_resources: bool = False
) -> str:
    """
    Save the current Blender file.

    Parameters:
    - filepath: Optional output .blend path (if omitted, saves current file)
    - pack_resources: Pack external resources before saving
    """
    try:
        blender = get_blender_connection()
        params = {"pack_resources": pack_resources}
        if filepath:
            params["filepath"] = filepath
        result = _handle_result(blender.send_command("save_blend_file", params))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error saving blend file: {str(e)}")
        return f"Error saving blend file: {str(e)}"


@mcp.tool()
def open_blend_file(ctx: Context, filepath: str) -> str:
    """
    Open a .blend file in Blender.

    Parameters:
    - filepath: Absolute path to the .blend file
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("open_blend_file", {
            "filepath": filepath
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error opening blend file: {str(e)}")
        return f"Error opening blend file: {str(e)}"


@mcp.tool()
def import_file(ctx: Context, filepath: str) -> str:
    """
    Import a 3D file into the current scene.

    Supported formats include: gltf/glb, fbx, obj, stl, ply, dae, abc, usd.

    Parameters:
    - filepath: Absolute path to file
    """
    try:
        blender = get_blender_connection()
        result = _handle_result(blender.send_command("import_file", {
            "filepath": filepath
        }))
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error importing file: {str(e)}")
        return f"Error importing file: {str(e)}"


@mcp.tool()
def search_blender_docs(
    ctx: Context,
    query: str,
    category: str = "all",
    max_results: int = 5
) -> str:
    """
    Search the Blender Python API documentation for reference information.
    Use this BEFORE writing execute_blender_code scripts to find correct API usage.

    The docs cover the full Blender 5.0 Python API: operators, types, bmesh,
    mathutils, GPU, freestyle, and best-practice guides.

    Parameters:
    - query: Search terms — can be multi-word. Examples:
             "bpy.ops.mesh.loopcut" — find a specific operator
             "boolean modifier" — find modifier docs
             "bmesh edge loop" — find bmesh edge operations
             "UV unwrap" — find UV mapping docs
             "shader nodes BSDF" — find node docs
             "particle system hair" — find particle docs
             "constraint copy rotation" — find constraint docs
    - category: Filter by doc section (narrows search, faster results):
                bpy_ops, bpy_types, guides, bmesh, mathutils, bpy_core,
                bpy_extras, bpy_app, gpu, freestyle, other, all
    - max_results: Maximum number of results to return (default 5, max 10)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("search_blender_docs", {
            "query": query,
            "category": category,
            "max_results": min(max(1, max_results), 10)
        })

        # In no-RAG setups, docs may not be mounted in Blender's addon path.
        # Degrade gracefully with a structured, non-throwing response.
        if isinstance(result, dict) and "error" in result:
            return json.dumps({
                "available": False,
                "message": result.get("error", "Documentation search unavailable."),
                "hint": "This does not impact core Blender MCP functionality. Set BLENDER_DOCS_PATH later if you want docs search.",
                "results": []
            }, indent=2)

        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error searching docs: {str(e)}")
        return json.dumps({
            "available": False,
            "message": f"Documentation search unavailable: {str(e)}",
            "hint": "Core Blender MCP tools continue to work without docs search.",
            "results": []
        }, indent=2)


# ============================================================================
# PROMPTS
# ============================================================================

@mcp.prompt()
def asset_creation_strategy() -> str:
    """Defines the preferred strategy for creating assets in Blender"""
    return """Use a simple, reliable workflow for Blender tasks:

        1) Session start
        - Run ping_blender() first.
        - Run get_mcp_capabilities() once to see enabled integrations.
        - Run get_scene_info() to understand current scene state.

        2) Preferred tools (simple and deterministic)
        - Use dedicated tools first: create_object, set_transform, add_modifier,
            create_material, assign_material, set_camera, add_light, set_keyframe,
            manage_collection, select_objects, frame_control, render_image, export_scene,
            save_blend_file, open_blend_file, import_file.
        - Use get_object_info() and get_viewport_screenshot() after major changes.

        3) Integrations (only if enabled)
        - PolyHaven: textures/HDRIs/models
        - Sketchfab: realistic downloadable models
        - Poly Pizza: low-poly downloadable models
        - GitHub Khronos: glTF sample models
        - Hyper3D / Hunyuan3D: custom single-item generation

        4) Advanced fallback
        - Use execute_blender_code() only when dedicated tools are insufficient.
        - Keep scripts short, step-by-step, and verify after each step.
        - search_blender_docs() is optional for advanced API lookup only.

        5) Quality checks before finishing
        - Confirm object transforms and world_bounding_box where relevant.
        - Ensure no clipping and correct spatial relationships.
        - Take a final viewport screenshot when useful.
        """


@mcp.prompt()
def llm_starter_system_prompt() -> str:
    """Starter prompt for LLMs using Blender MCP with deterministic tool order."""
    return """You are controlling Blender through Blender MCP.

Rules:
1) Always start with ping_blender(), then get_mcp_capabilities(), then get_scene_info().
2) Prefer dedicated tools over execute_blender_code().
3) After every major scene change, verify with get_object_info() or get_viewport_screenshot().
4) Use integrations only if enabled by get_mcp_capabilities().
5) Use search_blender_docs() only for advanced API lookup (optional, not required).
6) Keep actions small and deterministic: create/edit/verify/save.
7) Before finishing, run save_blend_file().

Default order for most tasks:
- inspect scene
- create or import objects
- transform and organize
- assign materials and lighting
- set camera
- optional animation
- render/export
- save
"""

# Main execution

def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()