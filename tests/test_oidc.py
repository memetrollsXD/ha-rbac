"""Tests for fail-closed OIDC role assignment."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.auth import EVENT_USER_ADDED
from homeassistant.auth.const import GROUP_ID_ADMIN, GROUP_ID_USER
from homeassistant.auth.models import Credentials, User, UserMeta
from homeassistant.auth.permissions.const import POLICY_CONTROL, POLICY_READ
from homeassistant.core import HomeAssistant

from custom_components.ha_rbac.const import ROLE_READ_ONLY
from custom_components.ha_rbac.oidc import OidcRoleAssigner, async_listen
from custom_components.ha_rbac.policy import Evaluator
from custom_components.ha_rbac.store import RbacStore


@pytest.fixture
async def setup(hass: HomeAssistant):
    """Create a real store, evaluator, and user-added listener."""
    await hass.auth.async_create_user("Owner", group_ids=[GROUP_ID_ADMIN])
    store = RbacStore(hass)
    await store.async_load()
    evaluator = Evaluator(hass, store)
    store.async_add_listener(evaluator.invalidate)
    assigner = OidcRoleAssigner(hass, store)
    unsubscribe = async_listen(hass, assigner)
    yield store, evaluator, assigner
    unsubscribe()


async def _student(store: RbacStore) -> dict:
    return await store.async_create_role(
        {
            "name": "Student",
            "allow": {"entities": {"entity_ids": {"light.classroom": True}}},
        }
    )


async def _new_oidc_user(hass: HomeAssistant) -> User:
    """Exercise Core's real credential-bearing user creation and event."""
    credential = Credentials("auth_oidc", None, {"sub": "subject"})
    provider = MagicMock()
    provider.async_credentials = AsyncMock(return_value=[])
    provider.async_user_meta_for_credentials = AsyncMock(
        return_value=UserMeta("Student", True, GROUP_ID_USER)
    )
    with patch.object(hass.auth, "_async_get_auth_provider", return_value=provider):
        return await hass.auth.async_get_or_create_user(credential)


async def _linked_oidc_user(hass: HomeAssistant) -> User:
    user = await hass.auth.async_create_user("Existing", group_ids=[GROUP_ID_USER])
    await hass.auth.async_link_user(
        user, Credentials("auth_oidc", None, {"sub": user.id})
    )
    return user


async def test_new_user_denied_before_event_then_assigned(
    hass: HomeAssistant, setup
) -> None:
    """The actual creation event never permits the old unrestricted fallback."""
    store, evaluator, _ = setup
    role = await _student(store)
    started = asyncio.Event()
    finish = asyncio.Event()
    save = store._async_save

    async def delayed_save() -> None:
        started.set()
        await finish.wait()
        await save()

    with patch.object(store, "_async_save", side_effect=delayed_save):
        user = await _new_oidc_user(hass)
        await started.wait()
        before = evaluator.async_permissions(user)
        assert not before.check_entity("light.classroom", POLICY_READ)
        assert not before.check_entity("lock.secret", POLICY_CONTROL)
        assert not before.full_access
        finish.set()
        await hass.async_block_till_done()
    assert store.bindings[user.id] == [role["id"]]
    after = evaluator.async_permissions(user)
    assert after is not before
    assert after.check_entity("light.classroom", POLICY_READ)
    assert not after.check_entity("lock.secret", POLICY_READ)


async def test_local_user_never_gets_student(hass: HomeAssistant, setup) -> None:
    """Local accounts remain unbound and denied."""
    store, evaluator, _ = setup
    await _student(store)
    user = await hass.auth.async_create_user("Local", group_ids=[GROUP_ID_USER])
    await hass.async_block_till_done()
    assert user.id not in store.bindings
    assert not evaluator.async_permissions(user).check_entity("light.a", POLICY_READ)


async def test_existing_linked_user_is_migrated_and_restart_is_idempotent(
    hass: HomeAssistant, setup
) -> None:
    """Existing users migrate once and the binding survives reload."""
    store, _, assigner = setup
    role = await _student(store)
    user = await _linked_oidc_user(hass)
    await hass.async_block_till_done()
    assert user.id not in store.bindings
    await assigner.async_scan()
    assert store.bindings[user.id] == [role["id"]]
    reloaded = RbacStore(hass)
    await reloaded.async_load()
    with patch.object(reloaded, "_async_save", wraps=reloaded._async_save) as save:
        await OidcRoleAssigner(hass, reloaded).async_scan()
        await OidcRoleAssigner(hass, reloaded).async_scan()
        save.assert_not_awaited()
    assert reloaded.bindings[user.id] == [role["id"]]


async def test_explicit_binding_is_preserved(hass: HomeAssistant, setup) -> None:
    """Automatic assignment never replaces an existing role."""
    store, _, assigner = setup
    await _student(store)
    user = await _linked_oidc_user(hass)
    await store.async_set_binding(user.id, [ROLE_READ_ONLY])
    await assigner.async_scan()
    await assigner.async_assign(user)
    assert store.bindings[user.id] == [ROLE_READ_ONLY]


@pytest.mark.parametrize("kind", ["owner", "system", "admin", "inactive_admin"])
async def test_privileged_users_are_excluded(
    hass: HomeAssistant, setup, kind: str
) -> None:
    """Owner, system, and administrator accounts retain their behavior."""
    store, evaluator, assigner = setup
    await _student(store)
    if kind == "owner":
        user = next(user for user in await hass.auth.async_get_users() if user.is_owner)
    elif kind == "system":
        user = await hass.auth.async_create_system_user("System")
    else:
        user = await hass.auth.async_create_user("Admin", group_ids=[GROUP_ID_ADMIN])
        if kind == "inactive_admin":
            await hass.auth.async_deactivate_user(user)
    await hass.auth.async_link_user(user, Credentials("auth_oidc", None, {"sub": kind}))
    await assigner.async_assign(user)
    assert user.id not in store.bindings
    if kind != "inactive_admin":
        assert evaluator.async_permissions(user).check_entity("lock.a", POLICY_READ)


async def test_missing_or_ambiguous_student_fails_closed(
    hass: HomeAssistant, setup, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing and ambiguous Student roles cannot grant access."""
    store, evaluator, assigner = setup
    user = await _linked_oidc_user(hass)
    await assigner.async_assign(user)
    assert "expected exactly one RBAC role named Student" in caplog.text
    assert not evaluator.async_permissions(user).check_entity("light.a", POLICY_READ)
    await _student(store)
    await _student(store)
    await assigner.async_assign(user)
    assert user.id not in store.bindings


async def test_duplicate_events_do_not_write_twice(hass: HomeAssistant, setup) -> None:
    """Concurrent duplicate events result in just one persistent write."""
    store, _, _ = setup
    await _student(store)
    user = await _linked_oidc_user(hass)
    await hass.async_block_till_done()
    with patch.object(store, "_async_save", wraps=store._async_save) as save:
        hass.bus.async_fire(EVENT_USER_ADDED, {"user_id": user.id})
        hass.bus.async_fire(EVENT_USER_ADDED, {"user_id": user.id})
        await hass.async_block_till_done()
        assert save.await_count == 1


async def test_empty_binding_denies_and_admin_fallback_remains(
    hass: HomeAssistant, setup
) -> None:
    """Missing and empty bindings grant no non-administrator access."""
    store, evaluator, _ = setup
    user = await hass.auth.async_create_user("Local", group_ids=[GROUP_ID_USER])
    for binding in (None, []):
        if binding is None:
            store.bindings.pop(user.id, None)
        else:
            store.bindings[user.id] = binding
        permissions = evaluator.async_permissions(user)
        assert not permissions.check_entity("lock.a", POLICY_READ)
        assert not permissions.tier_allowed("anything", "open")
        assert not permissions.full_access


async def test_assignment_failure_and_pending_write_fail_closed(
    hass: HomeAssistant, setup
) -> None:
    """Pending and failed writes cannot grant Student permissions."""
    store, evaluator, assigner = setup
    await _student(store)
    user = await _linked_oidc_user(hass)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def fail_save() -> None:
        started.set()
        await finish.wait()
        raise OSError("disk failure")

    with patch.object(store, "_async_save", side_effect=fail_save):
        task = asyncio.create_task(assigner.async_assign(user))
        await started.wait()
        assert not evaluator.async_permissions(user).check_entity(
            "light.classroom", POLICY_READ
        )
        finish.set()
        with pytest.raises(OSError, match="disk failure"):
            await task
    assert user.id not in store.bindings
    assert not evaluator.async_permissions(user).check_entity(
        "light.classroom", POLICY_READ
    )


async def test_manual_binding_wins_over_pending_assignment(
    hass: HomeAssistant, setup
) -> None:
    """A concurrent manual assignment is serialized after automatic assignment."""
    store, _, assigner = setup
    await _student(store)
    user = await _linked_oidc_user(hass)
    started = asyncio.Event()
    finish = asyncio.Event()
    save = store._async_save

    async def delayed_save() -> None:
        started.set()
        await finish.wait()
        await save()

    with patch.object(store, "_async_save", side_effect=delayed_save):
        automatic = asyncio.create_task(assigner.async_assign(user))
        await started.wait()
        manual = asyncio.create_task(store.async_set_binding(user.id, [ROLE_READ_ONLY]))
        finish.set()
        await asyncio.gather(automatic, manual)
    assert store.bindings[user.id] == [ROLE_READ_ONLY]
    reloaded = RbacStore(hass)
    await reloaded.async_load()
    assert reloaded.bindings[user.id] == [ROLE_READ_ONLY]


async def test_scan_continues_after_assignment_failure(
    hass: HomeAssistant, setup, caplog: pytest.LogCaptureFixture
) -> None:
    """A disk failure is logged and does not abort integration startup."""
    store, evaluator, assigner = setup
    await _student(store)
    user = await _linked_oidc_user(hass)
    with patch.object(store, "_async_save", side_effect=OSError("disk failure")):
        await assigner.async_scan()
    assert "Could not assign Student" in caplog.text
    assert not evaluator.async_permissions(user).check_entity(
        "light.classroom", POLICY_READ
    )


async def test_failed_assignment_is_not_persisted_by_another_store_write(
    hass: HomeAssistant, setup
) -> None:
    """Unrelated writes cannot save an assignment that subsequently fails."""
    store, _, assigner = setup
    await _student(store)
    user = await _linked_oidc_user(hass)
    started = asyncio.Event()
    finish = asyncio.Event()
    save = store._async_save
    calls = 0

    async def fail_first_save() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await finish.wait()
            raise OSError("disk failure")
        await save()

    with patch.object(store, "_async_save", side_effect=fail_first_save):
        automatic = asyncio.create_task(assigner.async_assign(user))
        await started.wait()
        other_write = asyncio.create_task(store.async_create_role({"name": "Other"}))
        finish.set()
        with pytest.raises(OSError, match="disk failure"):
            await automatic
        await other_write

    reloaded = RbacStore(hass)
    await reloaded.async_load()
    assert user.id not in reloaded.bindings
