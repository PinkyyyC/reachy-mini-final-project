"""Work around the SDK using an unreachable advertised Wi-Fi IP for WebRTC."""

from reachy_mini import ReachyMini
from reachy_mini.media.camera_constants import get_camera_specs_by_name
from reachy_mini.media.media_manager import MediaBackend, MediaManager


class NetworkReachyMini(ReachyMini):
    """Use the connected daemon's hostname for remote media as well as control.

    This overrides a private SDK hook in reachy-mini 1.14.0. Recheck it when
    upgrading the SDK; remove it once upstream supports a streaming host override.
    """

    def _configure_mediamanager(self, media_backend: str, log_level: str):
        if self.connection_mode != "network" or media_backend.lower() not in (
            "default", "auto", "webrtc"
        ):
            return super()._configure_mediamanager(media_backend, log_level)

        status = self.client.get_status()
        if getattr(status, "no_media", False):
            return super()._configure_mediamanager(media_backend, log_level)

        self._warn_if_daemon_version_mismatch(status)
        specs_name = getattr(status, "camera_specs_name", "")
        return MediaManager(
            backend=MediaBackend.WEBRTC,
            log_level=log_level,
            signalling_host=self.client.host,
            camera_specs=get_camera_specs_by_name(specs_name) if specs_name else None,
            daemon_url=self._daemon_http_url,
        )
