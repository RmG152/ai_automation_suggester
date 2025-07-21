# Storing Automation Requests in Files

This guide explains how to modify your Home Assistant automations to store recommendation data in files.

## Benefits of File Storage
- Persistent record of all suggestions
- Historical data analysis

## Step 1: Configure File Service

1. Go to **Settings > Devices & Services > + ADD INTEGRATION**
2. Search "File" from the list
3. Select "Set up a notification service"
4. Timestamp it's recommended

## Step 2: Add new Automation

```yaml
triggers:
  - trigger: state
    entity_id:
      - sensor.ai_automation_suggestions_google
    attribute: suggestions
conditions: []
actions:
  - action: notify.send_message
    metadata: {}
    data:
      message: >-
        {{ state_attr('sensor.ai_automation_suggestions_google', 'suggestions') }}
    target:
      entity_id: notify.file1
```

Multiple providers in the same automation:
```yaml
triggers:
  - entity_id: sensor.ai_automation_suggestions_google
    attribute: suggestions
    trigger: state
  - entity_id: sensor.ai_automation_suggestions_custom_openai
    attribute: suggestions
    trigger: state
  - entity_id: sensor.ai_automation_suggestions_openrouter
    attribute: suggestions
    trigger: state
  - entity_id: sensor.ai_automation_suggestions_generic_openai
    attribute: suggestions
    trigger: state
  - entity_id: sensor.ai_automation_suggestions_mistral_ai
    attribute: suggestions
    trigger: state
  - entity_id: sensor.ai_automation_suggestions_groq
    attribute: suggestions
    trigger: state
  - trigger: state
    entity_id:
      - sensor.ai_automation_suggestions_codestral
    attribute: suggestions
conditions:
  - condition: template
    value_template: "{{ trigger.to_state.attributes.suggestions is not none }}"
actions:
  - variables:
      provider_map:
        sensor.ai_automation_suggestions_google:
          counter: counter.gemini_crides_diaries
          file: notify.file_8
        sensor.ai_automation_suggestions_custom_openai:
          counter: counter.openrouter_crides_diaries
          file: notify.file_6
        sensor.ai_automation_suggestions_openrouter:
          counter: counter.openrouter_crides_diaries
          file: notify.file_10
        sensor.ai_automation_suggestions_generic_openai:
          counter: counter.openrouter_crides_diaries
          file: notify.file_7
        sensor.ai_automation_suggestions_mistral_ai:
          counter: counter.mistral_crides_diaries
          file: notify.file_9
        sensor.ai_automation_suggestions_groq:
          counter: counter.groq_crides_diaries
          file: notify.file_5
        sensor.ai_automation_suggestions_codestral:
          counter: counter.mistral_crides_diaries
          file: notify.file_11
      provider_info: "{{ provider_map[trigger.entity_id] }}"
  - action: notify.send_message
    metadata: {}
    data:
      message: "{{ trigger.to_state.attributes.suggestions | default('') }}"
    target:
      entity_id: "{{ provider_info.file }}"
  - target:
      entity_id: "{{ provider_info.counter }}"
    action: counter.increment
    data: {}
```

## File Path Configuration

| Location | Path Example | Use Case |
|----------|--------------|----------|
| Media directory | `/media/providerName_suggestions.md` | Persistent storage (recommended) |
| Add-on storage | `/share/providerName_suggestions.md` | Shared with other addons |

## Best Practices
1. Include timestamps in log entries
2. Rotate files periodically to prevent oversized logs
3. Use `.md` extension for better reading

## Troubleshooting
- **Permission errors**: Ensure Home Assistant has write access to the directory
- **File not updating**: Verify automation trigger conditions are met