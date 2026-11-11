"""Assign an existing role to eligible OIDC users."""

import logging

from homeassistant.auth import EVENT_USER_ADDED
from homeassistant.auth.const import GROUP_ID_ADMIN
from homeassistant.auth.models import User
from homeassistant.core import Event, HomeAssistant

from .store import RbacStore

_LOGGER = logging.getLogger(__name__)
OIDC_PROVIDER_TYPE = "auth_oidc"
STUDENT_ROLE = "Student"


class OidcRoleAssigner:
    """Assign the Student role without changing existing bindings."""

    def __init__(self, hass: HomeAssistant, store: RbacStore) -> None:
        """Store the dependencies."""
        self._hass = hass
        self._store = store

    async def async_user_added(self, event: Event) -> None:
        """Handle a newly created Home Assistant user."""
        user = await self._hass.auth.async_get_user(event.data["user_id"])
        if user is not None:
            await self.async_assign(user)

    async def async_scan(self) -> None:
        """Migrate existing eligible users."""
        for user in await self._hass.auth.async_get_users():
            try:
                await self.async_assign(user)
            except (OSError, ValueError):
                # async_assign logged the failure; keep enforcement running.
                continue

    async def async_assign(self, user: User) -> None:
        """Assign Student when the account is eligible and unbound."""
        if (
            user.is_owner
            or user.system_generated
            or user.is_admin
            or any(group.id == GROUP_ID_ADMIN for group in user.groups)
            or self._store.bindings.get(user.id)
            or not any(
                credential.auth_provider_type == OIDC_PROVIDER_TYPE
                for credential in user.credentials
            )
        ):
            return

        roles = [
            role_id
            for role_id, role in self._store.roles.items()
            if role.get("name") == STUDENT_ROLE
        ]
        if len(roles) != 1:
            _LOGGER.error(
                "OIDC user %s cannot be assigned: expected exactly one RBAC "
                "role named Student, found %s; the user remains unassigned",
                user.id,
                len(roles),
            )
            return
        try:
            await self._store.async_assign_if_unbound(user.id, roles[0], STUDENT_ROLE)
        except (OSError, ValueError):
            _LOGGER.exception("Could not assign Student to OIDC user %s", user.id)
            raise


def async_listen(hass: HomeAssistant, assigner: OidcRoleAssigner):
    """Listen for newly provisioned users."""
    return hass.bus.async_listen(EVENT_USER_ADDED, assigner.async_user_added)
