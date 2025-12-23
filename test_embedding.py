import time
import torch
from sentence_transformers import SentenceTransformer

# 1. Setup Sentences
sentences = [
    "Machine learning algorithms have revolutionized the way we process and analyze large datasets, enabling computers to learn patterns and make predictions without being explicitly programmed for every specific task.",
    "The Renaissance period in Europe, spanning roughly from the 14th to the 17th century, marked a profound cultural rebirth characterized by advances in art, science, literature, and philosophy that fundamentally changed Western civilization.",
    "Climate change poses one of the most significant challenges to humanity in the 21st century, with rising global temperatures leading to melting ice caps, extreme weather events, and disruptions to ecosystems worldwide.",
    "Quantum computing leverages the principles of quantum mechanics, such as superposition and entanglement, to perform computations that would be impossible or take thousands of years on classical computers.",
    "The human brain contains approximately 86 billion neurons that communicate through trillions of synaptic connections, creating the most complex biological structure known to science and enabling consciousness, memory, and thought.",
    "Sustainable agriculture practices, including crop rotation, integrated pest management, and reduced tillage, are essential for maintaining soil health, conserving water resources, and ensuring food security for future generations.",
    "The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine by providing the first widely effective antibiotic treatment, saving countless lives and paving the way for the development of modern pharmaceutical science.",
    "Deep learning neural networks, inspired by the structure and function of biological neural systems, have achieved remarkable success in tasks such as image recognition, natural language processing, and game playing.",
    "The Great Barrier Reef, located off the coast of Queensland, Australia, is the world's largest coral reef system, stretching over 2,300 kilometers and supporting an incredibly diverse ecosystem of marine life.",
    "Blockchain technology provides a decentralized and transparent method for recording transactions across multiple computers, ensuring data integrity and security without the need for a central authority or intermediary.",
    "The Industrial Revolution, beginning in Britain in the late 18th century, transformed economies from primarily agrarian to industrial and manufacturing-based, fundamentally changing labor patterns and social structures.",
    "Photosynthesis is the biochemical process by which plants, algae, and some bacteria convert light energy into chemical energy, producing oxygen as a byproduct and forming the foundation of most food chains on Earth.",
    "The theory of general relativity, formulated by Albert Einstein in 1915, describes gravity not as a force but as a curvature of spacetime caused by mass and energy, revolutionizing our understanding of the universe.",
    "Artificial neural networks consist of interconnected layers of nodes that process information through weighted connections, mimicking the way biological neurons transmit signals to learn complex patterns from data.",
    "The Amazon rainforest, often called the lungs of the Earth, produces approximately 20% of the world's oxygen and contains an estimated 10% of all species on the planet, making it crucial for global biodiversity.",
    "Gene editing technologies like CRISPR-Cas9 allow scientists to precisely modify DNA sequences in living organisms, opening unprecedented possibilities for treating genetic diseases and improving agricultural crops.",
    "The Internet of Things refers to the interconnected network of physical devices embedded with sensors, software, and connectivity that enables them to collect, exchange, and act on data without human intervention.",
    "Ancient Egyptian civilization, which flourished along the Nile River for over three thousand years, left behind remarkable achievements in architecture, mathematics, medicine, and governance that continue to fascinate us today.",
    "Renewable energy sources such as solar, wind, and hydroelectric power offer sustainable alternatives to fossil fuels, helping to reduce greenhouse gas emissions and combat climate change while meeting growing energy demands.",
    "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against pathogens, featuring both innate and adaptive responses that remember previous encounters with diseases.",
    "Cryptocurrency markets operate 24/7 on decentralized platforms, utilizing blockchain technology to enable peer-to-peer transactions without traditional banking intermediaries, though volatility and regulatory concerns remain significant challenges.",
    "The Apollo 11 mission in 1969 achieved humanity's first moon landing, with Neil Armstrong and Buzz Aldrin becoming the first humans to walk on another celestial body, marking a historic milestone in space exploration.",
    "Natural language processing combines computational linguistics with machine learning to enable computers to understand, interpret, and generate human language in ways that are both meaningful and contextually appropriate.",
    "The Mediterranean diet, rich in fruits, vegetables, whole grains, olive oil, and fish, has been associated with numerous health benefits including reduced risk of heart disease, stroke, and certain types of cancer.",
    "Plate tectonics theory explains how Earth's lithosphere is divided into large plates that move over the asthenosphere, causing earthquakes, volcanic activity, and the formation of mountains and ocean basins over geological time.",
    "Virtual reality technology creates immersive, computer-generated environments that users can interact with through specialized headsets and controllers, finding applications in gaming, education, training, and therapeutic treatments.",
    "The development of vaccines has been one of the most successful public health interventions in human history, preventing millions of deaths annually by training immune systems to recognize and fight specific pathogens.",
    "Urban planning involves the design and regulation of land use in cities and towns, balancing factors such as transportation infrastructure, housing density, green spaces, and economic development to create livable communities.",
    "String theory proposes that the fundamental constituents of the universe are not point particles but one-dimensional strings whose different vibration modes correspond to different particles, potentially unifying all forces of nature.",
    "The gut microbiome consists of trillions of microorganisms living in the human digestive tract, playing crucial roles in digestion, immune function, mental health, and overall wellbeing through complex interactions with their host.",
    "Cloud computing delivers computing services including storage, processing power, and applications over the internet, enabling businesses and individuals to access powerful resources on-demand without maintaining physical infrastructure.",
    "The French Revolution, beginning in 1789, overthrew the monarchy and aristocratic privileges, establishing principles of liberty, equality, and fraternity that profoundly influenced democratic movements and political thought worldwide.",
    "Neuroplasticity refers to the brain's remarkable ability to reorganize itself by forming new neural connections throughout life, allowing adaptation to learning, experience, and recovery from injury in ways previously thought impossible.",
    "The circular economy represents an alternative to the traditional linear take-make-dispose model, emphasizing the continuous reuse, recycling, and regeneration of materials to minimize waste and environmental impact.",
    "Gravitational waves, predicted by Einstein's theory of general relativity and first directly detected in 2015, are ripples in spacetime caused by massive accelerating objects, opening a new window for observing the universe.",
    "The Silk Road was an ancient network of trade routes connecting the East and West, facilitating not only the exchange of goods like silk, spices, and precious metals, but also ideas, technologies, and cultural practices.",
    "Autonomous vehicles use a combination of sensors, cameras, radar, and artificial intelligence to navigate roads without human intervention, promising to revolutionize transportation while raising questions about safety and ethics.",
    "The human genome project, completed in 2003, successfully mapped all approximately 3 billion base pairs of human DNA, providing unprecedented insights into genetic diseases, evolution, and what makes us uniquely human.",
    "Coral reefs are among the most biodiverse ecosystems on Earth, providing habitat for thousands of species, protecting coastlines from erosion, and supporting the livelihoods of millions of people, yet face severe threats from warming oceans.",
    "The printing press, invented by Johannes Gutenberg around 1440, revolutionized the dissemination of knowledge by making books affordable and accessible, contributing to the spread of literacy and the advancement of science and culture.",
    "Cybersecurity encompasses technologies, processes, and practices designed to protect networks, devices, programs, and data from attack, damage, or unauthorized access in an increasingly connected and vulnerable digital world.",
    "The water cycle, driven by solar energy, continuously moves water between Earth's surface and atmosphere through evaporation, condensation, precipitation, and collection, sustaining all life and regulating global climate patterns.",
    "Ancient Greek philosophy laid the foundations for Western thought through the works of Socrates, Plato, and Aristotle, who explored fundamental questions about ethics, knowledge, reality, and the nature of the good life.",
    "3D printing technology, also known as additive manufacturing, creates three-dimensional objects layer by layer from digital models, revolutionizing prototyping, customization, and production across industries from healthcare to aerospace.",
    "The concept of biodiversity encompasses the variety of life on Earth at all levels, from genes to ecosystems, providing essential services such as pollination, water purification, and climate regulation that support human civilization.",
    "Telemedicine leverages telecommunications technology to provide clinical healthcare services remotely, improving access to medical expertise especially in rural or underserved areas while reducing costs and travel burdens for patients.",
    "The Manhattan Project during World War II brought together leading scientists to develop the first nuclear weapons, demonstrating both the tremendous power of atomic energy and raising profound ethical questions about scientific responsibility.",
    "Epigenetics studies how environmental factors and behaviors can cause changes that affect gene expression without altering the underlying DNA sequence, potentially being passed down through generations and influencing health and disease.",
    "The concept of emotional intelligence, popularized in the 1990s, refers to the ability to recognize, understand, and manage one's own emotions while also being sensitive to others' emotions, proving crucial for personal and professional success.",
    "Nanotechnology manipulates matter at the molecular and atomic scale, creating materials and devices with novel properties that find applications in medicine, electronics, energy production, and environmental remediation with transformative potential.",
]


def run_benchmark(device_name):
    print(f"\n--- Running on {device_name.upper()} ---")

    # Load model to specific device
    try:
        model = SentenceTransformer("multi-qa-mpnet-base-cos-v1", device=device_name)
    except Exception as e:
        print(f"Could not load on {device_name}: {e}")
        return None

    # Warm-up (to ensure model is loaded into memory/cache)
    model.encode(sentences[:2])

    # Benchmark Start
    start_time = time.time()
    embeddings = model.encode(sentences)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Embedding shape: {embeddings.shape}")

    return embeddings, duration


# 2. Execute Benchmarks
# Check for Apple Silicon (MPS)
if torch.backends.mps.is_available():
    mps_embeddings, mps_time = run_benchmark("mps")
    cpu_embeddings, cpu_time = run_benchmark("cpu")

    speedup = cpu_time / mps_time
    print(f"\n🚀 MPS is {speedup:.2f}x faster than CPU on your Mac!")
else:
    print("MPS not available. Running on CPU only.")
    cpu_embeddings, cpu_time = run_benchmark("cpu")

# 3. Output/Inspect the actual data (using the last result)
final_embeddings = cpu_embeddings  # Using CPU as fallback
print("\n--- Sample Embedding (First 5 dimensions of Sentence 1) ---")
print(f"Sentence: '{sentences[0]}'")
print(f"Vector: {final_embeddings[0][:5]} ...")
