"""The AI Automation Suggester integration."""
import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import ConfigEntryNotReady, ServiceValidationError
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    PLATFORMS,
    CONF_PROVIDER,
    SERVICE_GENERATE_SUGGESTIONS,
    ATTR_PROVIDER_CONFIG,
    ATTR_CUSTOM_PROMPT,
    CONFIG_VERSION,
    INTEGRATION_NAME
)
from .coordinator import AIAutomationCoordinator

_LOGGER = logging.getLogger(__name__)

async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old config entry if necessary."""
    if config_entry.version < 2:
        _LOGGER.info("Migrating config entry from version 1 to 2")
        new_data = {**config_entry.data}
        key_mappings = {
            "openai_api_key": "api_key", "openai_model": "model", "openai_temperature": "temperature",
            "anthropic_api_key": "api_key", "anthropic_model": "model", "anthropic_temperature": "temperature",
            "google_api_key": "api_key", "google_model": "model", "google_temperature": "temperature",
            "groq_api_key": "api_key", "groq_model": "model", "groq_temperature": "temperature",
            "localai_model": "model", "localai_temperature": "temperature",
            "ollama_model": "model", "ollama_temperature": "temperature",
            "custom_openai_model": "model", "custom_openai_temperature": "temperature",
            "mistral_api_key": "api_key", "mistral_model": "model", "mistral_temperature": "temperature",
            "perplexity_api_key": "api_key", "perplexity_model": "model", "perplexity_temperature": "temperature",
            "openrouter_api_key": "api_key", "openrouter_model": "model", "openrouter_temperature": "temperature",
            "openai_azure_api_key": "api_key", "openai_azure_temperature": "temperature",
            "generic_openai_model": "model", "generic_openai_temperature": "temperature",
            "custom_openai_api_key": "api_key", "generic_openai_api_key": "api_key",
            "conf_codestral_api_key": "api_key", "conf_veniceai_api_key": "api_key",
            "conf_veniceai_temperature": "temperature", "conf_codestral_temperature": "temperature"
        }
        for old_key, new_key in key_mappings.items():
            if old_key in new_data:
                new_data[new_key] = new_data.pop(old_key)

        hass.config_entries.async_update_entry(config_entry, data=new_data, version=2)
        _LOGGER.info("Migration to version 2 successful")

    if config_entry.version < 3:
        _LOGGER.info("Migrating config entry from version 2 to 3 (sub-entries)")

        # Find if a main entry already exists, or create one
        main_entry = None
        for entry in hass.config_entries.async_entries(DOMAIN):
            if not entry.data.get(CONF_PROVIDER):
                main_entry = entry
                break

        if not main_entry:
            main_entry = hass.config_entries.async_create_entry(
                hass=hass,
                domain=DOMAIN,
                title=INTEGRATION_NAME,
                data={},
                version=3
            )

        # Migrate existing provider entries to sub-entries
        entries_to_migrate = [e for e in hass.config_entries.async_entries(DOMAIN) if e.data.get(CONF_PROVIDER)]
        for entry in entries_to_migrate:
            hass.config_entries.async_update_entry(
                entry,
                parent_entry_id=main_entry.entry_id,
                title=entry.data.get(CONF_PROVIDER, "Provider"),
                version=3
            )

        # After migration, we only need the main entry, the old ones become sub-entries
        # and don't need to be deleted explicitly as they are now part of the main entry.
        # However, we must update the main entry's version.
        hass.config_entries.async_update_entry(main_entry, version=3)

        _LOGGER.info("Migration to version 3 successful")

    return True


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the AI Automation Suggester component."""
    hass.data.setdefault(DOMAIN, {})

    async def handle_generate_suggestions(call: ServiceCall) -> None:
        """Handle the generate_suggestions service call."""
        provider_config = call.data.get(ATTR_PROVIDER_CONFIG)
        custom_prompt = call.data.get(ATTR_CUSTOM_PROMPT)
        all_entities = call.data.get("all_entities", False)
        domains = call.data.get("domains", {})
        entity_limit = call.data.get("entity_limit", 200)
        automation_read_yaml = call.data.get("automation_read_yaml", False)
        automation_limit = call.data.get("automation_limit", 100)
        include_entity_details = call.data.get("include_entity_details", True)

        if isinstance(domains, str):
            domains = [d.strip() for d in domains.split(',') if d.strip()]
        elif isinstance(domains, dict):
            domains = list(domains.keys())

        try:
            coordinator = None
            if provider_config:
                # Find coordinator by sub-entry ID
                for entry_id, coord in hass.data[DOMAIN].items():
                    if isinstance(coord, AIAutomationCoordinator) and coord.config_entry.entry_id == provider_config:
                        coordinator = coord
                        break
            else:
                for entry_id, coord in hass.data[DOMAIN].items():
                    if isinstance(coord, AIAutomationCoordinator):
                        coordinator = coord
                        break

            if coordinator is None:
                raise ServiceValidationError("No AI Automation Suggester provider configured")

            if custom_prompt:
                original_prompt = coordinator.SYSTEM_PROMPT
                coordinator.SYSTEM_PROMPT = f"{coordinator.SYSTEM_PROMPT}\n\nAdditional instructions:\n{custom_prompt}"
            else:
                original_prompt = None

            coordinator.scan_all = all_entities
            coordinator.selected_domains = domains
            coordinator.entity_limit = entity_limit
            coordinator.automation_read_file = automation_read_yaml
            coordinator.automation_limit = automation_limit
            coordinator.include_entity_details = include_entity_details

            try:
                await coordinator.async_request_refresh()
            finally:
                if original_prompt is not None:
                    coordinator.SYSTEM_PROMPT = original_prompt
                coordinator.scan_all = False
                coordinator.selected_domains = []
                coordinator.entity_limit = 200
                coordinator.automation_read_file = False
                coordinator.automation_limit = 100
                coordinator.include_entity_details = True

        except KeyError:
            raise ServiceValidationError("Provider configuration not found")
        except Exception as err:
            raise ServiceValidationError(f"Failed to generate suggestions: {err}")

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_SUGGESTIONS,
        handle_generate_suggestions
    )

    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up AI Automation Suggester from a config entry."""
    # This is the main entry, it doesn't have a provider itself.
    # We set up the platforms for sub-entries.
    if not entry.data.get(CONF_PROVIDER):
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
        entry.async_on_unload(entry.add_update_listener(async_reload_entry))
        return True

    # This is a sub-entry (a provider)
    try:
        coordinator = AIAutomationCoordinator(hass, entry)
        hass.data[DOMAIN][entry.entry_id] = coordinator

        _LOGGER.debug(
            "Setup complete for %s with provider %s",
            entry.title,
            entry.data.get(CONF_PROVIDER)
        )

        @callback
        def handle_custom_event(event):
            _LOGGER.debug("Received custom event '%s', triggering suggestions with all_entities=True", event.event_type)
            hass.async_create_task(coordinator_request_all_suggestions())

        async def coordinator_request_all_suggestions():
            coordinator.scan_all = True
            await coordinator.async_request_refresh()
            coordinator.scan_all = False

        entry.async_on_unload(hass.bus.async_listen("ai_automation_suggester_update", handle_custom_event))

        return True

    except Exception as err:
        _LOGGER.error("Failed to setup integration: %s", err)
        raise ConfigEntryNotReady from err

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # If this is the main entry, unload sub-entries as well
    if not entry.data.get(CONF_PROVIDER):
        return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    # This is a sub-entry
    try:
        if entry.entry_id in hass.data[DOMAIN]:
            coordinator = hass.data[DOMAIN].pop(entry.entry_id)
            await coordinator.async_shutdown()
        return True
    except Exception as err:
        _LOGGER.error("Error unloading entry: %s", err)
        return False

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
