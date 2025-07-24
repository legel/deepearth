"""
Configuration for different DeepSeek embedding options
"""

# DeepSeek model options
DEEPSEEK_MODELS = {
    # Smaller, faster models
    'deepseek-coder-1.3b': {
        'model_name': 'deepseek-ai/deepseek-coder-1.3b-base',
        'embedding_dim': 2048,
        'description': 'Smaller DeepSeek model, good for code and technical text'
    },
    
    # Larger, more powerful models
    'deepseek-coder-6.7b': {
        'model_name': 'deepseek-ai/deepseek-coder-6.7b-base',
        'embedding_dim': 4096,
        'description': 'Larger DeepSeek model, better semantic understanding'
    },
    
    # Instruction-tuned models
    'deepseek-coder-1.3b-instruct': {
        'model_name': 'deepseek-ai/deepseek-coder-1.3b-instruct',
        'embedding_dim': 2048,
        'description': 'Instruction-tuned model, good for descriptions'
    },
    
    # Alternative: Use sentence transformers (much lighter)
    'sentence-transformer': {
        'model_name': 'all-MiniLM-L6-v2',
        'embedding_dim': 384,
        'description': 'Lightweight alternative using sentence-transformers'
    },
    
    # Alternative: Use OpenAI embeddings (if you have API key)
    'openai-embedding': {
        'model_name': 'text-embedding-ada-002',
        'embedding_dim': 1536,
        'description': 'OpenAI embeddings (requires API key)'
    }
}

# Rich species descriptions for better embeddings
SPECIES_DESCRIPTIONS = {
    'Helianthus debilis': {
        'common_name': 'Beach Sunflower',
        'description': 'A coastal flowering plant with bright yellow petals resembling small sunflowers. Native to Florida beaches and dunes, highly salt-tolerant perennial with sprawling growth habit. Blooms year-round in warm climates.',
        'habitat': 'Coastal dunes, beaches, salt marshes',
        'characteristics': 'Yellow flowers, sprawling habit, salt-tolerant, drought-resistant'
    },
    
    'Gaillardia pulchella': {
        'common_name': 'Blanket Flower',
        'description': 'Vibrant wildflower with red and yellow daisy-like blooms. Petals are typically red with yellow tips. Drought-tolerant and thrives in sandy soils and coastal areas. Attracts butterflies and other pollinators.',
        'habitat': 'Sandy soils, coastal plains, roadsides',
        'characteristics': 'Red and yellow flowers, drought-tolerant, full sun, attracts butterflies'
    },
    
    'Iris virginica': {
        'common_name': 'Blue Flag Iris',
        'description': 'Wetland plant with striking blue-purple iris flowers. Found in marshes, swamps, and pond edges throughout Florida. Sword-like leaves grow in fans. Important for wetland restoration.',
        'habitat': 'Wetlands, marshes, pond edges, swamps',
        'characteristics': 'Blue-purple flowers, sword-like leaves, wetland plant, spring bloomer'
    },
    
    'Lonicera sempervirens': {
        'common_name': 'Coral Honeysuckle',
        'description': 'Native vine with tubular red-orange flowers that attract hummingbirds. Semi-evergreen climber that blooms spring through fall. Unlike invasive Japanese honeysuckle, this native species is well-behaved.',
        'habitat': 'Forest edges, fencerows, gardens',
        'characteristics': 'Red tubular flowers, vine, attracts hummingbirds, semi-evergreen'
    },
    
    'Hamelia patens': {
        'common_name': 'Firebush',
        'description': 'Tropical shrub with orange-red tubular flowers blooming year-round. Important nectar source for butterflies and hummingbirds. Produces small black berries eaten by birds. Heat and drought tolerant.',
        'habitat': 'Hammocks, gardens, forest edges',
        'characteristics': 'Orange-red flowers, attracts wildlife, fast-growing, tropical'
    },
    
    'Echinacea purpurea': {
        'common_name': 'Purple Coneflower',
        'description': 'Medicinal plant with distinctive pink-purple petals and raised central cone. Native to eastern North America, attracts butterflies and goldfinches. Used in herbal medicine for immune support.',
        'habitat': 'Prairies, open woodlands, gardens',
        'characteristics': 'Pink-purple flowers, medicinal, attracts butterflies, drought-tolerant'
    },
    
    'Serenoa repens': {
        'common_name': 'Saw Palmetto',
        'description': 'Low-growing palm with fan-shaped leaves and sharply serrated petioles. Produces fragrant white flowers and dark berries. Important food source for wildlife. Used medicinally for prostate health.',
        'habitat': 'Pine flatwoods, coastal dunes, hammocks',
        'characteristics': 'Fan palm, serrated stems, berries, slow-growing, hardy'
    },
    
    'Hymenocallis latifolia': {
        'common_name': 'Spider Lily',
        'description': 'Bulbous plant with exotic white flowers featuring long, thin, spider-like petals. Blooms in summer with fragrant flowers. Prefers moist conditions. Leaves are strap-like and emerge from bulbs.',
        'habitat': 'Wetlands, marshes, moist hammocks',
        'characteristics': 'White spider-like flowers, fragrant, bulbous, summer bloomer'
    },
    
    'Psychotria nervosa': {
        'common_name': 'Wild Coffee',
        'description': 'Understory shrub with glossy dark green leaves showing prominent veins. Small white flowers in clusters followed by bright red berries. Not true coffee but related. Important food for birds.',
        'habitat': 'Hammocks, understory, shade gardens',
        'characteristics': 'Glossy leaves, white flowers, red berries, shade-tolerant'
    },
    
    'Canna flaccida': {
        'common_name': 'Yellow Canna',
        'description': 'Wetland plant with large banana-like leaves and showy yellow orchid-like flowers. Native aquatic canna found in marshes and swamps. Flowers attract pollinators and seeds feed waterfowl.',
        'habitat': 'Wetlands, marshes, pond edges, swamps',
        'characteristics': 'Yellow flowers, large leaves, aquatic, attracts pollinators'
    }
}


def create_rich_description(species_name: str, include_all: bool = True) -> str:
    """Create a rich description for DeepSeek embedding"""
    if species_name in SPECIES_DESCRIPTIONS:
        info = SPECIES_DESCRIPTIONS[species_name]
        
        if include_all:
            # Full rich description
            description = f"""
Species: {species_name}
Common Name: {info['common_name']}
Description: {info['description']}
Habitat: {info['habitat']}
Characteristics: {info['characteristics']}
"""
        else:
            # Shorter description
            description = f"{info['common_name']}: {info['description']}"
            
        return description.strip()
    else:
        # Fallback for unknown species
        common_name = species_name.replace('_', ' ').title()
        return f"Plant species: {species_name} ({common_name}). Native Florida plant."


def get_embedding_prompt_template(style: str = 'botanical') -> str:
    """Get prompt template for creating embeddings"""
    templates = {
        'botanical': "Botanical species {species}: {description}",
        'ecological': "In the ecosystem: {species} - {description} Habitat: {habitat}",
        'identification': "Plant identification - Species: {species}, Common name: {common_name}, Key features: {characteristics}",
        'simple': "{common_name} ({species})"
    }
    
    return templates.get(style, templates['botanical'])
