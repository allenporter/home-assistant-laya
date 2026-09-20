"""Config flow for laya integration."""

from __future__ import annotations

from typing import Any
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_COMPOUND_THRESHOLD,
    CONF_CONFIDENCE_THRESHOLD,
    CONF_DEVICE,
    CONF_FALLBACK_AGENT,
    DEFAULT_COMPOUND_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DOMAIN,
)


class LayaConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Laya."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is not None:
            return self.async_create_entry(title="Laya", data=user_input)

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_DEVICE, default=DEFAULT_DEVICE
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["auto", "cpu", "cuda", "mps"],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(step_id="user", data_schema=schema)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return LayaOptionsFlowHandler()


class LayaOptionsFlowHandler(OptionsFlow):
    """Handle options for Laya."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_COMPOUND_THRESHOLD,
                    default=self.config_entry.options.get(
                        CONF_COMPOUND_THRESHOLD, DEFAULT_COMPOUND_THRESHOLD
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Optional(
                    CONF_CONFIDENCE_THRESHOLD,
                    default=self.config_entry.options.get(
                        CONF_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        mode=selector.NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Optional(
                    CONF_FALLBACK_AGENT,
                    description={
                        "suggested_value": self.config_entry.options.get(
                            CONF_FALLBACK_AGENT
                        )
                    },
                ): selector.ConversationAgentSelector(),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
