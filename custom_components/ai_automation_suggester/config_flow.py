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

    VERSION = 1

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
                return await {
                    "OpenAI": self.async_step_openai,
                    "Anthropic": self.async_step_anthropic,
                    "Google": self.async_step_google,
                    "Groq": self.async_step_groq,
                    "LocalAI": self.async_step_localai,
                    "Ollama": self.async_step_ollama,
                    "Custom OpenAI": self.async_step_custom_openai,
                    "Mistral AI": self.async_step_mistral,
                    "Perplexity AI": self.async_step_perplexity,
                    "OpenRouter": self.async_step_openrouter,
                    "OpenAI Azure": self.async_step_openai_azure,
                    "Generic OpenAI": self.async_step_generic_openai,
                }[self.provider]()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_PROVIDER): vol.In(
                        [
                            "Anthropic",
                            "Custom OpenAI",
                            "Generic OpenAI",
                            "Google",
                            "Groq",
                            "LocalAI",
                            "Mistral AI",
                            "Ollama",
                            "OpenAI Azure",
                            "OpenAI",
                            "OpenRouter",
                            "Perplexity AI",
                        ]
                    )
                }
            ),
            errors=errors,
        )

    # ───────── helper to reduce repetition ─────────
    async def _provider_form(
        self,
        step_id: str,
        schema: vol.Schema,
        validate_fn,
        user_input: Dict[str, Any] | None,
    ):
        errors: Dict[str, str] = {}
        placeholders: Dict[str, str] = {}

        if user_input:
            self.validator = ProviderValidator(self.hass)
            err = await validate_fn(user_input)
            if err is None:
                self.data.update(user_input)
                return await self.async_step_common()
            errors["base"] = "api_error"
            placeholders["error_message"] = err

        return self.async_show_form(step_id=step_id, data_schema=schema, errors=errors, description_placeholders=placeholders)

    # ───────── provider‑specific steps (OpenAI shown; others similar) ─────────
    async def async_step_common(self, user_input=None):
        """Handle the common configuration options."""
        if user_input is not None:
            self.data.update(user_input)
            title = f"AI Automation Suggester ({self.provider})"
            return self.async_create_entry(title=title, data=self.data)

        schema = vol.Schema({
            vol.Required(CONF_API_KEY): TextSelector(TextSelectorConfig(type="password")),
            vol.Optional(CONF_MODEL, default=DEFAULT_MODELS.get(self.provider)): str,
            vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(CONF_MAX_INPUT_TOKENS, default=DEFAULT_MAX_INPUT_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=100)
            ),
            vol.Optional(CONF_MAX_OUTPUT_TOKENS, default=DEFAULT_MAX_OUTPUT_TOKENS): vol.All(
                vol.Coerce(int), vol.Range(min=100)
            ),
        })

        return self.async_show_form(step_id="common", data_schema=schema)


    async def async_step_openai(self, user_input=None):
        return await self._provider_form(
            "openai",
            vol.Schema({}),
            lambda ui: self.validator.validate_openai(ui[CONF_API_KEY]),
            user_input,
        )

    async def async_step_anthropic(self, user_input=None):
        async def _v(ui):
            return await self.validator.validate_anthropic(
                ui[CONF_API_KEY], ui.get(CONF_MODEL, DEFAULT_MODELS["Anthropic"])
            )

        return await self._provider_form(
            "anthropic",
            vol.Schema({}),
            _v,
            user_input,
        )

    async def async_step_google(self, user_input=None):
        async def _v(ui):
            return await self.validator.validate_google(
                ui[CONF_API_KEY], ui.get(CONF_MODEL, DEFAULT_MODELS["Google"])
            )

        return await self._provider_form(
            "google",
            vol.Schema({}),
            _v,
            user_input,
        )

    async def async_step_groq(self, user_input=None):
        return await self._provider_form(
            "groq",
            vol.Schema({}),
            lambda ui: self.validator.validate_groq(ui[CONF_API_KEY]),
            user_input,
        )

    async def async_step_localai(self, user_input=None):
        async def _v(ui):
            return await self.validator.validate_localai(ui[CONF_LOCALAI_IP_ADDRESS], ui[CONF_LOCALAI_PORT], ui[CONF_LOCALAI_HTTPS])

        schema = vol.Schema({
            vol.Required(CONF_LOCALAI_IP_ADDRESS): str,
            vol.Required(CONF_LOCALAI_PORT, default=8080): int,
            vol.Required(CONF_LOCALAI_HTTPS, default=False): bool,
        })
        return await self._provider_form(
            "localai", 
            schema,
            _v,
            user_input,
        )

    async def async_step_ollama(self, user_input=None):
        async def _v(ui):
            return await self.validator.validate_ollama(ui[CONF_OLLAMA_IP_ADDRESS], ui[CONF_OLLAMA_PORT], ui[CONF_OLLAMA_HTTPS])

        schema = vol.Schema({
            vol.Required(CONF_OLLAMA_IP_ADDRESS): str,
            vol.Required(CONF_OLLAMA_PORT, default=11434): int,
            vol.Required(CONF_OLLAMA_HTTPS, default=False): bool,
            vol.Optional(CONF_OLLAMA_DISABLE_THINK, default=False): bool,
   
        })
        return await self._provider_form(
            "ollama",
            schema,
            _v,
            user_input,
        )

    async def async_step_custom_openai(self, user_input=None):
        async def _v(ui):
            return await self.validator.validate_custom_openai(ui[CONF_CUSTOM_OPENAI_ENDPOINT], ui.get(CONF_CUSTOM_OPENAI_API_KEY))

        schema = vol.Schema({
            vol.Required(CONF_CUSTOM_OPENAI_ENDPOINT): str,
            vol.Optional(CONF_CUSTOM_OPENAI_API_KEY): TextSelector(TextSelectorConfig(type="password")),
        })
        return await self._provider_form(
            "custom_openai",
            schema,
            _v,
            user_input,
        )

    # Mistral: no live validation needed
    async def async_step_mistral(self, user_input=None):
        if user_input:
            self.data.update(user_input)
            return await self.async_step_common(user_input=None)

        return self.async_show_form(step_id="mistral", data_schema=vol.Schema({}))

    async def async_step_perplexity(self, user_input=None):
        async def _v(ui):
            return await self.validator.validate_perplexity(
                ui[CONF_API_KEY], ui.get(CONF_MODEL, DEFAULT_MODELS["Perplexity AI"])
            )

        return await self._provider_form(
            "perplexity",
            vol.Schema({}),
            _v,
            user_input,
        )

    async def async_step_openrouter(self, user_input=None):
        async def _v(ui):
            return await self.validator.validate_openrouter(
                ui[CONF_API_KEY],
                ui.get(CONF_MODEL, DEFAULT_MODELS["OpenRouter"]),
            )

        schema = vol.Schema({
            vol.Optional(CONF_OPENROUTER_REASONING_MAX_TOKENS, default=0): vol.All(
                vol.Coerce(int), vol.Range(min=0)
            ),
        })
        return await self._provider_form(
            "openrouter",
            schema,
            _v,
            user_input,
        )

    async def async_step_openai_azure(self, user_input=None):
        async def _v(ui):
            if not ui.get(CONF_API_KEY) or not ui.get(CONF_OPENAI_AZURE_DEPLOYMENT_ID) or not ui.get(CONF_OPENAI_AZURE_API_VERSION):
                return "All fields are required"
            return None

        schema = vol.Schema({
            vol.Required(CONF_OPENAI_AZURE_DEPLOYMENT_ID, default=DEFAULT_MODELS["OpenAI Azure"]): str,
            vol.Required(CONF_OPENAI_AZURE_ENDPOINT, default="{your-resource-name}.openai.azure.com"): str,
            vol.Required(CONF_OPENAI_AZURE_API_VERSION, default="2025-01-01-preview"): str,
        })
        return await self._provider_form(
            "openai_azure",
            schema,
            _v,
            user_input,
        )

    async def async_step_generic_openai(self, user_input=None):
        """Handle the Generic OpenAI API configuration."""
        async def _v(ui):
            if not ui.get(CONF_GENERIC_OPENAI_ENDPOINT):
                return "API URL is required"
            if ui.get(CONF_GENERIC_OPENAI_ENABLE_VALIDATION, False):
                if not ui.get(CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT):
                    return "Validation endpoint is required when validation is enabled"
                return await self.validator.validate_generic_openai(ui[CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT], ui.get(CONF_GENERIC_OPENAI_API_KEY))

        schema = vol.Schema({
            vol.Required(CONF_GENERIC_OPENAI_ENDPOINT): str,
            vol.Optional(CONF_GENERIC_OPENAI_API_KEY): TextSelector(TextSelectorConfig(type="password")),
            vol.Optional(CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT, default=""): str,
            vol.Optional(CONF_GENERIC_OPENAI_ENABLE_VALIDATION, default=False): bool,
        })
        return await self._provider_form(
            "generic_openai",
            schema,
            _v,
            user_input,
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
        }

        # provider‑specific editable fields
        if provider == "OpenAI":
            schema[vol.Optional(CONF_API_KEY, default=self._get_option(CONF_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["OpenAI"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "Anthropic":
            schema[vol.Optional(CONF_API_KEY, default=self._get_option(CONF_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["Anthropic"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "Google":
            schema[vol.Optional(CONF_API_KEY, default=self._get_option(CONF_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["Google"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "Groq":
            schema[vol.Optional(CONF_API_KEY, default=self._get_option(CONF_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["Groq"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "LocalAI":
            schema[vol.Optional(CONF_LOCALAI_HTTPS, default=self._get_option(CONF_LOCALAI_HTTPS, False))] = bool
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["LocalAI"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
            schema[vol.Optional(CONF_LOCALAI_IP_ADDRESS, default=self._get_option(CONF_LOCALAI_IP_ADDRESS, "localhost"))] = str
            schema[vol.Optional(CONF_LOCALAI_PORT, default=self._get_option(CONF_LOCALAI_PORT, 8080))] = int
        elif provider == "Ollama":
            schema[vol.Optional(CONF_OLLAMA_IP_ADDRESS, default=self._get_option(CONF_OLLAMA_IP_ADDRESS, "localhost"))] = str
            schema[vol.Optional(CONF_OLLAMA_PORT, default=self._get_option(CONF_OLLAMA_PORT, 11434))] = int
            schema[vol.Optional(CONF_OLLAMA_HTTPS, default=self._get_option(CONF_OLLAMA_HTTPS, False))] = bool
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["Ollama"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
            schema[vol.Optional(CONF_OLLAMA_DISABLE_THINK, default=self._get_option(CONF_OLLAMA_DISABLE_THINK, False))] = bool    
        elif provider == "Custom OpenAI":
            schema[vol.Optional(CONF_CUSTOM_OPENAI_ENDPOINT, default=self._get_option(CONF_CUSTOM_OPENAI_ENDPOINT))] = str
            schema[vol.Optional(CONF_CUSTOM_OPENAI_API_KEY, default=self._get_option(CONF_CUSTOM_OPENAI_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["Custom OpenAI"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "Mistral AI":
            schema[vol.Optional(CONF_API_KEY, default=self._get_option(CONF_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["Mistral AI"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "Perplexity AI":
            schema[vol.Optional(CONF_API_KEY, default=self._get_option(CONF_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["Perplexity AI"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "OpenRouter":
            schema[vol.Optional(CONF_API_KEY, default=self._get_option(CONF_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["OpenRouter"]))] = str
            schema[vol.Optional(CONF_OPENROUTER_REASONING_MAX_TOKENS, default=self._get_option(CONF_OPENROUTER_REASONING_MAX_TOKENS, 0))] = vol.All(vol.Coerce(int), vol.Range(min=0))
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "OpenAI Azure":
            schema[vol.Optional(CONF_API_KEY, default=self._get_option(CONF_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_OPENAI_AZURE_ENDPOINT, default=self._get_option(CONF_OPENAI_AZURE_ENDPOINT))] = str
            schema[vol.Optional(CONF_OPENAI_AZURE_DEPLOYMENT_ID, default=self._get_option(CONF_OPENAI_AZURE_DEPLOYMENT_ID, DEFAULT_MODELS["OpenAI Azure"]))] = str
            schema[vol.Optional(CONF_OPENAI_AZURE_API_VERSION, default=self._get_option(CONF_OPENAI_AZURE_API_VERSION, "2025-01-01-preview"))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
        elif provider == "Generic OpenAI":
            schema[vol.Optional(CONF_GENERIC_OPENAI_API_KEY, default=self._get_option(CONF_GENERIC_OPENAI_API_KEY))] = TextSelector(TextSelectorConfig(type="password"))
            schema[vol.Optional(CONF_GENERIC_OPENAI_ENDPOINT, default=self._get_option(CONF_GENERIC_OPENAI_ENDPOINT))] = str
            schema[vol.Optional(CONF_MODEL, default=self._get_option(CONF_MODEL, DEFAULT_MODELS["Generic OpenAI"]))] = str
            schema[vol.Optional(CONF_TEMPERATURE, default=self._get_option(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))] = vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
            schema[vol.Optional(CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT, default=self._get_option(CONF_GENERIC_OPENAI_VALIDATION_ENDPOINT, ""))] = str
            schema[vol.Optional(CONF_GENERIC_OPENAI_ENABLE_VALIDATION, default=self._get_option(CONF_GENERIC_OPENAI_ENABLE_VALIDATION, False))] = bool

        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))
