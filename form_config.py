form_config = {
    'structured_yield_state': {
        'category_title': 'yield',
        'unit': 'kg/ha/year',
        'fields': [
            {
                'field_title': 'corn production (wet basis)',
                'unit': 'kg/ha/year',
                'custom': False,
                'inputSource': 'ai',
                'value': '11612',
                'reference': '',
                'additionalNotes': '',
                'required': True,
                'dropdownValues': None
            },
            {
                'field_title': 'moisture content',
                'unit': '%',
                'custom': False,
                'inputSource': 'ai',
                'value': '15.5',
                'reference': '',
                'additionalNotes': '',
                'required': True,
                'dropdownValues': None
            },
            {
                'field_title': 'fraction of crop area burned',
                'unit': '',
                'custom': False,
                'inputSource': 'user',
                'value': '',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            },
            {
                'field_title': 'straw yield removed',
                'unit': '',
                'custom': False,
                'inputSource': 'user',
                'value': '',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            },
            {
                'field_title': 'soil condition',
                'unit': '',
                'custom': False,
                'inputSource': 'user',
                'value': '',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            }
        ]
    },
    'structured_liquid_fuels_consumption_state': {
        'category_title': 'liquid fuels consumption',
        'unit': 'unknown',
        'fields': [
            {
                'field_title': 'diesel',
                'unit': 'gallons/acre',
                'custom': False,
                'inputSource': 'user',
                'value': '6.07596',
                'reference': '',
                'additionalNotes': '',
                'required': True,
                'dropdownValues': None
            },
            {
                'field_title': 'gasoline',
                'unit': 'gallons/acre',
                'custom': False,
                'inputSource': 'user',
                'value': '0',
                'reference': '',
                'additionalNotes': '',
                'required': True,
                'dropdownValues': None
            },
            {
                'field_title': 'heavy fuel oil',
                'unit': 'unknown',
                'custom': True,
                'inputSource': 'user',
                'value': 'not provided',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            },
            {
                'field_title': 'methanol',
                'unit': 'unknown',
                'custom': True,
                'inputSource': 'user',
                'value': 'not provided',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            },
            {
                'field_title': 'ethanol',
                'unit': 'liters/year',
                'custom': True,
                'inputSource': 'user',
                'value': '245094297',
                'reference': '',
                'additionalNotes': '',
                'required': True,
                'dropdownValues': None
            }
        ]
    },
    'structured_gaseous_fuels_consumption_state': {
        'category_title': 'gaseous fuels consumption',
        'unit': '',
        'fields': [
            {
                'field_title': 'lpg',
                'unit': '',
                'custom': False,
                'inputSource': 'ai',
                'value': '',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            },
            {
                'field_title': 'natural gas on higher heating value (hhv) basis: ',
                'unit': '',
                'custom': False,
                'inputSource': 'ai',
                'value': '',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            },
            {
                'field_title': 'biogas',
                'unit': '',
                'custom': False,
                'inputSource': 'ai',
                'value': '',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            },
            {
                'field_title': 'hydrogen',
                'unit': '',
                'custom': False,
                'inputSource': 'ai',
                'value': '',
                'reference': '',
                'additionalNotes': '',
                'required': False,
                'dropdownValues': None
            }
        ]
    }
}