import json
from form_config import form_config

def title_case(s: str) -> str:
    """Convert snake_case to Title Case"""
    return ' '.join(word.capitalize() for word in s.split('_'))

def clean_state_name(state_name: str) -> str:
    """Remove structured_ prefix and _state suffix, convert to Title Case, and strip spaces"""
    name = state_name
    if name.startswith('structured_'):
        name = name[10:]  # Remove 'structured_' prefix
    if name.endswith('_state'):
        name = name[:-6]  # Remove '_state' suffix
    return title_case(name).strip()

def format_field_title(field_title: str) -> str:
    """Format field title with first word in Title Case and rest in lowercase"""
    words = field_title.split()
    if not words:
        return field_title
    return f"{words[0].title()} {' '.join(word.lower() for word in words[1:])}"

# Define the correct category order based on reference
CATEGORY_ORDER = [
    "Corn Yield",
    "Liquid Fuels Consumption",
    "Gaseous Fuels Consumption",
    "Solid Fuels Consumption",
    "Electricity Consumption",
    "Synthetic Nitrogen Fertilizer Consumption",
    "Organic Nitrogen Fertilizer Consumption",
    "P₂O₅ Fertilizers Consumption",
    "K₂O Fertilizers Consumption",
    "CaO Fertilizers Consumption",
    "Other Fertilizers Consumption",
    "Plant Protection Products Consumption",
    "Seeding Materials Consumption",
    "Land Use Change Emissions",
    "Emissions Saving From Soil Carbon Accumulation",
    "Corn Transport and Distribution"
]

def format_field(field: dict) -> dict:
    """Format a single field according to the new schema"""
    formatted_field = {
        'field_title': format_field_title(field['field_title']),
        'unit': field['unit'],
        'custom': field['custom'],
        'inputSource': 'manual' if field['inputSource'] == 'user' else field['inputSource'],
        'additionalNotes': field.get('additionalNotes', ''),
        'required': field.get('required', False),
    }
    
    # Add inputType if needed
    if field.get('dropdownValues'):
        formatted_field['inputType'] = 'dropdown'
        formatted_field['dropdown'] = 'Yes'
        formatted_field['dropdownValues'] = field['dropdownValues']
    else:
        formatted_field['inputType'] = 'number'
    
    return formatted_field

def sort_formatted_config(formatted_config: list) -> list:
    """
    Sort the formatted config according to predefined category order.
    Categories not in the predefined order will be added at the end.
    """
    # Create a dictionary for quick lookup of indices with stripped categories
    order_dict = {cat.lower().strip(): i for i, cat in enumerate(CATEGORY_ORDER)}
    
    def get_sort_key(item):
        category = item['category_title'].lower().strip()
        return order_dict.get(category, float('inf'))
    
    return sorted(formatted_config, key=get_sort_key)

def format_config_for_calculator(config: dict) -> list:
    """Transform the form config into the calculator format"""
    formatted_states = {}
    
    # First format all states
    for state_name, state_data in config.items():
        category_title = clean_state_name(state_name)
        formatted_state = {
            'category_title': category_title,
            'unit': state_data['unit'],
            'fields': [format_field(field) for field in state_data['fields']]
        }
        
        # Add additional properties based on category
        if 'fuel' in state_name.lower():
            formatted_state.update({
                'addCustomFieldButton': True,
                'addCustomFieldButtonText': 'Add',
                'addCustomFieldButtonDescription': f"Add another {category_title.lower()}",
                'dropDownOptions': ['Others (Specify)'],
                'defaultFieldCount': 3
            })
        
        # Special case for transport
        if 'transport' in state_name.lower():
            formatted_state.update({
                'subtitle': 'Leg 1',
                'addCustomFieldButton': False,
                'addAnotherLegButton': True,
                'addAnotherLegButtonText': 'Add Another Leg',
                'removeLegButton': False,
                'replicable': True
            })
        
        formatted_states[category_title] = formatted_state
    
    # Convert to list and sort
    formatted_config = list(formatted_states.values())
    return sort_formatted_config(formatted_config)

def save_formatted_config(formatted_config: list, output_file: str = 'formatted_config.json'):
    """Save the formatted config to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(formatted_config, f, indent=2)

if __name__ == '__main__':
    # Format the config
    formatted_config = format_config_for_calculator(form_config)
    
    # Save to file
    save_formatted_config(formatted_config)