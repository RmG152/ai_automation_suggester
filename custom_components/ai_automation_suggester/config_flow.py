# custom_components/ai_automation_suggester/config_flow.py
"""Config flow for AI Automation Suggester."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import TextSelector, TextSelectorConfig

from .const import *

_LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Lightweight provider validators (unchanged)
# ─────────────────────────────────────────────────────────────
class ProviderValidator:
    """Ping each provider with a dummy request to validate credentials."""

    def __init__(self, hass):
        self.session = async_get_clientsession(hass)

    async def validate_openai(self, api_key: str) -> Optional[str]:
        hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            resp = await self.session.get("https://api.openai.com/v1/models", headers=hdr)
            return None if resp.status == 200 else await resp.text()
        except Exception as err:  # noqa: BLE001
            return str(err)

    async def validate_anthropic(self, api_key: str, model: str) -> Optional[str]:
        hdr = {
            "x-api-key": api_key,
            "anthropic-version": VERSION_ANTHROPIC,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
            "max_tokens": 1,
        }
        try:
            resp = await self.session.post("https://api.anthropic.com/v1/messages", headers=hdr, json=payload)
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_google(self, api_key: str, model: str) -> Optional[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 1}}
        try:
            resp = await self.session.post(url, json=payload)
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_groq(self, api_key: str) -> Optional[str]:
        hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            resp = await self.session.get("https://api.groq.com/openai/v1/models", headers=hdr)
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_localai(self, ip: str, port: int, https: bool) -> Optional[str]:
        proto = "https" if https else "http"
        try:
            resp = await self.session.get(f"{proto}://{ip}:{port}/v1/models")
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_ollama(self, ip: str, port: int, https: bool) -> Optional[str]:
        proto = "https" if https else "http"
        try:
            resp = await self.session.get(f"{proto}://{ip}:{port}/api/tags")
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_openwebui(self, ip: str, port: int, https: bool, api_key: str) -> Optional[str]:
        hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        proto = "https" if https else "http"
        try:
            resp = await self.session.get(f"{proto}://{ip}:{port}/api/tags")
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_custom_openai(self, endpoint: str, api_key: str | None) -> Optional[str]:
        hdr = {"Content-Type": "application/json"}
        if api_key:
            hdr["Authorization"] = f"Bearer {api_key}"
        try:
            resp = await self.session.get(f"{endpoint}/v1/models", headers=hdr)
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_perplexity(self, api_key: str, model: str) -> Optional[str]:
        hdr = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
        try:
            resp = await self.session.post(ENDPOINT_PERPLEXITY, headers=hdr, json=payload)
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_openrouter(self, api_key: str, model: str) -> Optional[str]:
        hdr = {"content-type": "application/json"}
        if api_key:
            hdr["Authorization"] = f"Bearer {api_key}"
        try:
            resp = await self.session.get(
                "https://openrouter.ai/api/v1/models", headers=hdr
            )
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)

    async def validate_generic_openai(self, endpoint: str, api_key: str) -> Optional[str]:
        hdr = {"Content-Type": "application/json"}
        if api_key:
            hdr["Authorization"] = f"Bearer {api_key}"
        try:
            resp = await self.session.get(f"{endpoint}", headers=hdr)
            return None if resp.status == 200 else await resp.text()
        except Exception as err:
            return str(err)


# ─────────────────────────────────────────────────────────────
# Config‑flow main class
# ─────────────────────────────────────────────────────────────
class AIAutomationConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle integration setup via the UI."""

    VERSION = 2

    def __init__(self) -> None:
        self.provider: str | None = None
        self.data: Dict[str, Any] = {}
        self.validator: ProviderValidator | None = None

    # ───────── Initial provider choice ─────────
    async def async_step_user(self, user_input: Dict[str, Any] | None = None):
        errors: Dict[str, str] = {}
        if user_input:
            self.provider = user_input[CONF_PROVIDER]
            self.data.update(user_input)

            if any(ent.data.get(CONF_PROVIDER) == self.provider for ent in self._async_current_entries()):
                errors["base"] = "already_configured"
            else:
                return await self.async_step_provider_config()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_PROVIDER): vol.In(
                        [
                            "Anthropic",
                            "Custom OpenAI",
                            "Codestral",
                            "Generic OpenAI",
                            "Google",
                            "Groq",
                            "LocalAI",
                            "Mistral AI",
                            "Ollama",
                            "OpenAI Azure",
                            "OpenAI",
                            "OpenRouter",
                            "Open Web UI",
                            "Perplexity AI",
                            "Venice AI",
                        ]
                    )
                }
            ),
            errors=errors,
        )

    # ───────── Common configuration step for all providers ─────────
    async def async_step_provider_config(self, user_input: Dict[str, Any] | None = None):
        """Handle the provider configuration options."""
        errors: Dict[str, str] = {}
        
        if user_input is not None:
            # Basic validation
            if not user_input.get(CONF_API_KEY) and self.provider not in ["LocalAI", "Ollama", "Open Web UI"]:
                errors["base"] = "api_key_required"
            
            # Initialize validator if needed
            if not self.validator:
                self.validator = ProviderValidator(self.hass)
                
            # Provider-specific validation
            if not errors:
                error_msg = None
                try:
                    if self.provider == "OpenAI":
                        error_msg = await self.validator.validate_openai(user_input[CONF_API_KEY])
                    elif self.provider == "Anthropic":
                        error_msg = await self.validator.validate_anthropic(
                            user_input[CONF_API_KEY], 
                            user_input.get(CONF_MODEL, DEFAULT_MODELS["Anthropic"])
                        )
                    elif self.provider == "Google":
                        error_msg = await self.validator.validate_google(
                            user_input[CONF_API_KEY],
                            user_input.get(CONF_MODEL, DEFAULT_MODELS["Google"])
                        )
                    elif self.provider == "Groq":
                        error_msg = await self.validator.validate_groq(user_input[CONF_API_KEY])
                    elif self.provider == "LocalAI":
                        error_msg = await self.validator.validate_localai(
                            user_input.get(CONF_LOCALAI_IP_ADDRESS, "localhost"),
                            user_input.get(CONF_LOCALAI_PORT, 8080),
                            user_input.get(CONF_LOCALAI_HTTPS, False)
                        )
                    elif self.provider == "Ollama":
                        error_msg = await self.validator.validate_ollama(
                            user_input.get(CONF_OLLAMA_IP_ADDRESS, "localhost"),
                            user_input.get(CONF_OLLAMA_PORT, 11434),
                            user_input.get(CONF_OLLAMA_HTTPS, False)
                        )
                    elif self.provider == "Open Web UI":
                        error_msg = await self.validator.validate_openwebui(
                            user_input.get(CONF_OPENWEBUI_IP_ADDRESS, "localhost"),
                            user_input.get(CONF_OPENWEBUI_PORT, 11434),
                            user_input.get(CONF_OPENWEBUI_HTTPS, False),
                            user_input.get[CONF_API_KEY]
                        )   
                    elif self.provider == "Custom OpenAI":
                        error_msg = await self.validator.validate_custom_openai(
                            user_input.get(CONF_CUSTOM_OPENAI_ENDPOINT, ""),
                            user_input.get(CONF_API_KEY)
                        )
                    elif self.provider == "Perplexity AI":
                        error_msg = await self.validator.validate_perplexity(
                            user_input[CONF_API_KEY],
                            user_input.get(CONF_MODEL, DEFAULT_MODELS["Perplexity AI"])
                        )
                    elif self.provider == "OpenRouter":
                        error_msg = await self.validator.validate_openrouter(
                            user_input[CONF_API_KEY],
                            user_input.get(CONF_MODEL, DEFAULT_MODELS["OpenRouter"])
                        )
                    elif self.provider == "Generic OpenAI":
                        if user_input.get(CONF_GENERIC_OPENAI_ENABLE_VALIDATION):
                            error_msg = await self.validator.validate_generic_openai(
                                user_input.get(CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT, ""),
                                user_input[CONF_API_KEY]
                            )

                    if error_msg:
                        errors["base"] = "auth_error"
                        _LOGGER.error("Validation error for %s: %s", self.provider, error_msg)

                except Exception as ex:
                    errors["base"] = "unknown"
                    _LOGGER.exception("Unexpected error validating %s: %s", self.provider, ex)

            if not errors:
                self.data.update(user_input)
                title = f"AI Automation Suggester ({self.provider})"
                return self.async_create_entry(title=title, data=self.data)

        # Build dynamic schema based on provider capabilities
        schema_dict = {
            vol.Optional(CONF_API_KEY): TextSelector(TextSelectorConfig(type="password")),
            vol.Optional(CONF_MODEL, default=DEFAULT_MODELS.get(self.provider, "")): str,
            vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.All(
                vol.Coerce(float), vol.Range(min=0.0, max=2.0)
            ),
            vol.Optional(CONF_MAX_INPUT_TOKENS, default=DEFAULT_MAX_INPUT_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=100)
            ),
            vol.Optional(CONF_MAX_OUTPUT_TOKENS, default=DEFAULT_MAX_OUTPUT_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=100)
            ),
            vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.All(
                vol.Coerce(int), vol.Range(min=15)
            ),
        }

        # Add provider-specific fields
        if self.provider == "LocalAI":
            schema_dict.update({
                vol.Optional(CONF_LOCALAI_HTTPS, default=False): bool,
                vol.Optional(CONF_LOCALAI_IP_ADDRESS, default="localhost"): str,
                vol.Optional(CONF_LOCALAI_PORT, default=8080): int,
            })
        elif self.provider == "Ollama":
            schema_dict.update({
                vol.Optional(CONF_OLLAMA_IP_ADDRESS, default="localhost"): str,
                vol.Optional(CONF_OLLAMA_PORT, default=11434): int,
                vol.Optional(CONF_OLLAMA_HTTPS, default=False): bool,
                vol.Optional(CONF_OLLAMA_DISABLE_THINK, default=False): bool,
            })
        elif self.provider == "Open Web UI":
            schema_dict.update({
                vol.Optional(CONF_OPENWEBUI_IP_ADDRESS, default="localhost"): str,
                vol.Optional(CONF_OPENWEBUI_PORT, default=11434): int,
                vol.Optional(CONF_OPENWEBUI_HTTPS, default=False): bool,
                vol.Optional(CONF_OPENWEBUI_DISABLE_THINK, default=False): bool,
            })            
        elif self.provider == "Custom OpenAI":
            schema_dict[vol.Optional(CONF_CUSTOM_OPENAI_ENDPOINT)] = str
        elif self.provider == "OpenRouter":
            schema_dict[vol.Optional(CONF_OPENROUTER_REASONING_MAX_TOKENS, default=0)] = vol.All(vol.Coerce(int), vol.Range(min=0))
        elif self.provider == "OpenAI Azure":
            schema_dict.update({
                vol.Optional(CONF_OPENAI_AZURE_ENDPOINT): str,
                vol.Optional(CONF_OPENAI_AZURE_DEPLOYMENT_ID, default=DEFAULT_MODELS["OpenAI Azure"]): str,
                vol.Optional(CONF_OPENAI_AZURE_API_VERSION, default="2025-01-01-preview"): str,
            })
        elif self.provider == "Generic OpenAI":
            schema_dict.update({
                vol.Optional(CONF_GENERIC_OPENAI_ENDPOINT): str,
                vol.Optional(CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT, default=""): str,
                vol.Optional(CONF_GENERIC_OPENAI_ENABLE_VALIDATION, default=False): bool,
            })
        elif self.provider == "Google":
            schema_dict.update({
                vol.Optional(CONF_GOOGLE_THINKING_MODE, default="default"): vol.In(["default", "custom", "disabled"]),
                vol.Optional(CONF_GOOGLE_THINKING_BUDGET, default="-1"): vol.All(vol.Coerce(int), vol.Range(min=-1, max=32768)),
                vol.Optional(CONF_GOOGLE_ENABLE_SEARCH, default=False): bool,
            })


        return self.async_show_form(
            step_id="provider_config",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
            description_placeholders={"provider": self.provider}
        )

    # ───────── Options flow (edit after setup) ─────────
    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return AIAutomationOptionsFlowHandler(config_entry)


class AIAutomationOptionsFlowHandler(config_entries.OptionsFlow):
    """Allow post‑setup tweaking of models, keys, token budgets."""

    def __init__(self, config_entry):
        super().__init__()
        self._config_entry = config_entry

    def _get_option(self, key, default=None):
        """Get value from options, then data, then default."""
        if key in self._config_entry.options:
            return self._config_entry.options.get(key)
        if key in self._config_entry.data:
            return self._config_entry.data.get(key)
        return default

    async def async_step_init(self, user_input=None):
        if user_input:
            new_data = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=new_data)

        provider = self._config_entry.data.get(CONF_PROVIDER)
        schema: Dict[Any, Any] = {
            vol.Optional(
                CONF_MAX_INPUT_TOKENS,
                default=self._get_option(CONF_MAX_INPUT_TOKENS, DEFAULT_MAX_INPUT_TOKENS)
            ): vol.All(vol.Coerce(int), vol.Range(min=100)),
            vol.Optional(
                CONF_MAX_OUTPUT_TOKENS,
                default=self._get_option(CONF_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS)
            ): vol.All(vol.Coerce(int), vol.Range(min=100)),
            vol.Required(CONF_API_KEY, default=self._get_option(CONF_API_KEY)): TextSelector(TextSelectorConfig(type="password")),
            vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS[provider])): str,
            vol.Optional(CONF_TIMEOUT, default=self._get_option(CONF_TIMEOUT, DEFAULT_TIMEOUT)): vol.All(vol.Coerce(int), vol.Range(min=15)),
        }

        # provider‑specific editable fields
        if provider == "LocalAI":
            schema[vol.Optional(CONF_LOCALAI_HTTPS, default=self._get_option(CONF_LOCALAI_HTTPS, False))] = bool
            schema[vol.Optional(CONF_LOCALAI_IP_ADDRESS, default=self._get_option(CONF_LOCALAI_IP_ADDRESS, "localhost"))] = str
            schema[vol.Optional(CONF_LOCALAI_PORT, default=self._get_option(CONF_LOCALAI_PORT, 8080))] = int
        elif provider == "Ollama":
            schema[vol.Optional(CONF_OLLAMA_IP_ADDRESS, default=self._get_option(CONF_OLLAMA_IP_ADDRESS, "localhost"))] = str
            schema[vol.Optional(CONF_OLLAMA_PORT, default=self._get_option(CONF_OLLAMA_PORT, 11434))] = int
            schema[vol.Optional(CONF_OLLAMA_HTTPS, default=self._get_option(CONF_OLLAMA_HTTPS, False))] = bool
            schema[vol.Optional(CONF_OLLAMA_DISABLE_THINK, default=self._get_option(CONF_OLLAMA_DISABLE_THINK, False))] = bool
        elif provider == "Open Web UI":
            schema[vol.Optional(CONF_OPENWEBUI_IP_ADDRESS, default=self._get_option(CONF_OPENWEBUI_IP_ADDRESS, "localhost"))] = str
            schema[vol.Optional(CONF_OPENWEBUI_PORT, default=self._get_option(CONF_OPENWEBUI_PORT, 11434))] = int
            schema[vol.Optional(CONF_OPENWEBUI_HTTPS, default=self._get_option(CONF_OPENWEBUI_HTTPS, False))] = bool
            schema[vol.Optional(CONF_OPENWEBUI_DISABLE_THINK, default=self._get_option(CONF_OPENWEBUI_DISABLE_THINK, False))] = bool
        elif provider == "Custom OpenAI":
            schema[vol.Optional(CONF_CUSTOM_OPENAI_ENDPOINT, default=self._get_option(CONF_CUSTOM_OPENAI_ENDPOINT))] = str
        elif provider == "OpenRouter":
            schema[vol.Optional(CONF_OPENROUTER_REASONING_MAX_TOKENS, default=self._get_option(CONF_OPENROUTER_REASONING_MAX_TOKENS, 0))] = vol.All(vol.Coerce(int), vol.Range(min=0))
        elif provider == "OpenAI Azure":
            schema[vol.Optional(CONF_OPENAI_AZURE_ENDPOINT, default=self._get_option(CONF_OPENAI_AZURE_ENDPOINT))] = str
            schema[vol.Optional(CONF_OPENAI_AZURE_DEPLOYMENT_ID, default=self._get_option(CONF_OPENAI_AZURE_DEPLOYMENT_ID, DEFAULT_MODELS["OpenAI Azure"]))] = str
            schema[vol.Optional(CONF_OPENAI_AZURE_API_VERSION, default=self._get_option(CONF_OPENAI_AZURE_API_VERSION, "2025-01-01-preview"))] = str
        elif provider == "Generic OpenAI":
            schema[vol.Optional(CONF_GENERIC_OPENAI_ENDPOINT, default=self._get_option(CONF_GENERIC_OPENAI_ENDPOINT))] = str
            schema[vol.Optional(CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT, default=self._get_option(CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT, ""))] = str
            schema[vol.Optional(CONF_GENERIC_OPENAI_ENABLE_VALIDATION, default=self._get_option(CONF_GENERIC_OPENAI_ENABLE_VALIDATION, False))] = bool
        elif provider == "Google":
            schema[vol.Optional(CONF_GOOGLE_THINKING_MODE, default=self._get_option(CONF_GOOGLE_THINKING_MODE, "default"))] = vol.In(["default", "custom", "disabled"])
            schema[vol.Optional(CONF_GOOGLE_THINKING_BUDGET, default=self._get_option(CONF_GOOGLE_THINKING_BUDGET, "-1"))] = vol.All(vol.Coerce(int), vol.Range(min=-1, max=32768))
            schema[vol.Optional(CONF_GOOGLE_ENABLE_SEARCH, default=self._get_option(CONF_GOOGLE_ENABLE_SEARCH, False))] = bool

        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))
