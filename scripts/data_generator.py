import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_airline_data(num_records=30000):
    """Generates synthetic airline pricing and delay dataset."""
    
    # Base configuration
    airlines_config = {
        'Emirates': {'base_price_mult': 1.2, 'delay_prob_base': 0.08},
        'Qatar Airways': {'base_price_mult': 1.15, 'delay_prob_base': 0.09},
        'Etihad': {'base_price_mult': 1.1, 'delay_prob_base': 0.10},
        'Flydubai': {'base_price_mult': 0.7, 'delay_prob_base': 0.20},
        'Air Arabia': {'base_price_mult': 0.65, 'delay_prob_base': 0.22},
        'Oman Air': {'base_price_mult': 1.0, 'delay_prob_base': 0.12},
    }
    airlines = list(airlines_config.keys())

    cities = ['Dubai', 'Doha', 'Abu Dhabi', 'Muscat', 'Riyadh', 'Jeddah', 'Kuwait City', 'Manama']
    
    # Define realistic route distances/durations (approximate in minutes)
    route_durations = {}
    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if src != dst:
                base_duration = 60 + abs(i - j) * 30 + random.randint(-15, 15)
                route_durations[(src, dst)] = base_duration

    departure_times = ['Morning', 'Afternoon', 'Evening', 'Night']
    classes = ['Economy', 'Business']
    stops_options = ['Non-stop', '1 stop', '2+ stops']

    data = []

    for _ in range(num_records):
        airline = random.choice(airlines)
        source = random.choice(cities)
        dest = random.choice([c for c in cities if c != source])
        
        dept_time = random.choice(departure_times)
        travel_class = random.choice(classes)
        stops = random.choices(stops_options, weights=[0.6, 0.3, 0.1])[0]
        days_left = random.randint(1, 45)
        
        base_dur = route_durations[(source, dest)]
        
        # Adjust duration based on stops
        if stops == '1 stop':
            duration = base_dur + random.randint(45, 120)
        elif stops == '2+ stops':
            duration = base_dur + random.randint(120, 240)
        else:
            duration = base_dur
            
        # --- PRICING LOGIC ---
        # Base price depends on duration
        base_price = duration * 1.5
        
        # Airline multiplier
        base_price *= airlines_config[airline]['base_price_mult']
        
        # Class multiplier
        if travel_class == 'Business':
            base_price *= random.uniform(3.5, 5.0)
            
        # Stops multiplier (non-stop is more expensive)
        if stops == 'Non-stop':
            base_price *= 1.3
        elif stops == '1 stop':
            base_price *= 1.0
        else:
            base_price *= 0.8
            
        # Time of day (peak hours morning/evening are more expensive)
        if dept_time in ['Morning', 'Evening']:
            base_price *= 1.2
        elif dept_time == 'Night':
            base_price *= 0.85
            
        # Days left (prices increase exponentially as days_left approaches 0)
        if days_left <= 7:
            days_mult = 1 + (7 - days_left) * 0.15 # Up to +105%
        elif days_left <= 14:
            days_mult = 1.1
        else:
            days_mult = 1.0 - (days_left - 14) * 0.005 # Slow decay earlier
        
        price = base_price * days_mult
        
        # Add some noise
        price *= random.uniform(0.95, 1.05)
        price = round(price, 2)
        
        # --- DELAY LOGIC ---
        delay_prob = airlines_config[airline]['delay_prob_base']
        
        # Time of day delays (cascading effect)
        if dept_time == 'Afternoon':
            delay_prob += 0.05
        elif dept_time == 'Evening':
            delay_prob += 0.10
        elif dept_time == 'Night':
            delay_prob += 0.15
            
        # Stops delay (more stops = higher chance of delay)
        if stops == '1 stop':
            delay_prob += 0.12
        elif stops == '2+ stops':
            delay_prob += 0.25
            
        # Add random randomness
        is_delayed = 1 if random.random() < delay_prob else 0
        
        data.append({
            'airline': airline,
            'source_city': source,
            'destination_city': dest,
            'departure_time': dept_time,
            'duration': duration,
            'days_left': days_left,
            'class': travel_class,
            'stops': stops,
            'price': price,
            'delay': is_delayed
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic airline dataset...")
    df = generate_airline_data(35000)
    
    os.makedirs('data', exist_ok=True)
    file_path = 'data/airline_data.csv'
    df.to_csv(file_path, index=False)
    
    print(f"Dataset generated with {len(df)} rows.")
    print("\nSample Data:")
    print(df.head())
    
    print("\nClass vs median price:")
    print(df.groupby('class')['price'].median())
    
    print("\nStops vs mean delay probability:")
    print(df.groupby('stops')['delay'].mean())
