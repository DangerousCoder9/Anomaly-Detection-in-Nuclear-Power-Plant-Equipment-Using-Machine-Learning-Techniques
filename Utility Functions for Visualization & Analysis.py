"""
Feature Descriptions for Nuclear Power Plant Monitoring

This file documents the 8 sensor features used in the anomaly detection system.
"""

FEATURE_DESCRIPTIONS = {
    # Reactor Core Parameters (3 features)
    'Reactor_Core_Temperature': {
        'description': 'Measures the thermal condition of the reactor fuel region',
        'unit': 'Celsius',
        'category': 'Reactor Core',
        'anomaly_indicator': 'Abnormal spikes may indicate cooling failure or power imbalance',
        'normal_range': (300, 350)
    },
    
    'Neutron_Flux': {
        'description': 'Represents the intensity of neutron activity',
        'unit': 'neutrons/cm²/s',
        'category': 'Reactor Core',
        'anomaly_indicator': 'Sudden increase or oscillation can indicate instability',
        'normal_range': (1e12, 1e14)
    },
    
    'Reactor_Pressure': {
        'description': 'Monitors internal reactor pressure',
        'unit': 'MPa',
        'category': 'Reactor Core',
        'anomaly_indicator': 'Unexpected drops or rises may signal coolant system issues',
        'normal_range': (15.0, 16.0)
    },
    
    # Coolant System Parameters (2 features)
    'Coolant_Inlet_Temperature': {
        'description': 'Temperature of coolant entering the reactor core',
        'unit': 'Celsius',
        'category': 'Coolant System',
        'anomaly_indicator': 'Higher-than-normal values indicate insufficient heat removal',
        'normal_range': (280, 290)
    },
    
    'Coolant_Flow_Rate': {
        'description': 'Measures the volume of coolant circulating through the system',
        'unit': 'm³/s',
        'category': 'Coolant System',
        'anomaly_indicator': 'Reduced flow can lead to overheating conditions',
        'normal_range': (40, 50)
    },
    
    # Turbine / Power Output Parameters (2 features)
    'Turbine_Speed': {
        'description': 'Rotational speed of the steam turbine',
        'unit': 'RPM',
        'category': 'Turbine / Power Output',
        'anomaly_indicator': 'Instability or sudden drops may indicate mechanical faults',
        'normal_range': (1400, 1500)
    },
    
    'Electrical_Power_Output': {
        'description': 'Amount of electricity generated',
        'unit': 'MW',
        'category': 'Turbine / Power Output',
        'anomaly_indicator': 'Unexpected deviation from expected output indicates system imbalance',
        'normal_range': (800, 900)
    },
    
    # Safety Monitoring (1 feature)
    'Radiation_Level': {
        'description': 'Monitors radiation intensity in containment areas',
        'unit': 'Sv/h',
        'category': 'Safety Monitoring',
        'anomaly_indicator': 'Increase beyond normal baseline is strong anomaly indicator',
        'normal_range': (0.01, 0.05)
    }
}

SENSOR_CATEGORIES = {
    'Reactor Core': [
        'Reactor_Core_Temperature',
        'Neutron_Flux',
        'Reactor_Pressure'
    ],
    'Coolant System': [
        'Coolant_Inlet_Temperature',
        'Coolant_Flow_Rate'
    ],
    'Turbine / Power Output': [
        'Turbine_Speed',
        'Electrical_Power_Output'
    ],
    'Safety Monitoring': [
        'Radiation_Level'
    ]
}

def print_feature_info():
    """Print detailed feature information"""
    print("="*80)
    print("NUCLEAR POWER PLANT SENSOR FEATURES")
    print("="*80)
    
    for category, features in SENSOR_CATEGORIES.items():
        print(f"\n{category}:")
        print("-" * 80)
        
        for feature_name in features:
            info = FEATURE_DESCRIPTIONS[feature_name]
            print(f"\n  {feature_name}")
            print(f"    Description: {info['description']}")
            print(f"    Unit: {info['unit']}")
            print(f"    Normal Range: {info['normal_range']}")
            print(f"    Anomaly Indicator: {info['anomaly_indicator']}")


if __name__ == "__main__":
    print_feature_info()