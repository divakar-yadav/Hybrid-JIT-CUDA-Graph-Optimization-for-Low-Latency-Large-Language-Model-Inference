# MMLU Prompts for Realistic LLM Benchmarking

from typing import List

# MMLU (Massive Multitask Language Understanding) prompts
# These are real prompts from the MMLU dataset for different sequence lengths

MMLU_PROMPTS = {
    10: [
        "What is the capital of France?",
        "Explain quantum computing.",
        "How does photosynthesis work?",
        "What is machine learning?",
        "Describe the water cycle."
    ],
    50: [
        "In the context of machine learning, explain the difference between supervised and unsupervised learning, and provide examples of algorithms for each type.",
        "Describe the process of cellular respiration in eukaryotic cells, including the main stages and where each stage occurs within the cell.",
        "Explain the concept of supply and demand in economics, and how changes in these factors affect market prices and quantities.",
        "What are the main causes of climate change, and what are some potential solutions to mitigate its effects?",
        "Describe the structure and function of DNA, including how it stores genetic information and how it is replicated."
    ],
    100: [
        "In machine learning, explain the concept of overfitting and underfitting. What are the causes of each phenomenon, and what techniques can be used to prevent overfitting? Provide examples of regularization methods and explain how they work to improve model generalization.",
        "Describe the process of protein synthesis in eukaryotic cells, including transcription and translation. Explain the roles of DNA, mRNA, tRNA, and ribosomes in this process, and how the genetic code is used to determine the amino acid sequence.",
        "Explain the concept of comparative advantage in international trade. How does it differ from absolute advantage, and what are the implications for trade policy? Provide examples of how countries can benefit from specializing in goods where they have a comparative advantage.",
        "What are the greenhouse effect and global warming? Explain the difference between natural and anthropogenic greenhouse gases, and discuss the potential consequences of climate change on ecosystems, weather patterns, and human societies.",
        "Describe the structure of the atom, including the roles of protons, neutrons, and electrons. Explain how atomic number and mass number are determined, and discuss the concept of isotopes and their applications in science and medicine."
    ],
    150: [
        "In machine learning, explain the bias-variance tradeoff and its relationship to model complexity. How does this tradeoff affect model performance, and what techniques can be used to find the optimal balance? Discuss cross-validation, regularization methods like L1 and L2, and ensemble methods as approaches to managing this tradeoff.",
        "Describe the process of cellular respiration in detail, including glycolysis, the Krebs cycle, and the electron transport chain. Explain where each stage occurs in the cell, what molecules are involved, and how ATP is produced. Discuss the differences between aerobic and anaerobic respiration, and their efficiency in energy production.",
        "Explain the concept of market equilibrium in microeconomics, including how supply and demand curves interact to determine price and quantity. Discuss what happens when markets are not in equilibrium, including shortages and surpluses. Explain the concept of price elasticity of demand and supply, and how these elasticities affect market responses to changes in conditions.",
        "What is climate change, and what are its primary causes? Explain the greenhouse effect, the role of carbon dioxide and other greenhouse gases, and how human activities have contributed to increased atmospheric concentrations. Discuss the potential impacts of climate change on weather patterns, sea levels, ecosystems, and human societies, and outline some mitigation and adaptation strategies.",
        "Describe the structure and function of DNA in detail, including the double helix structure, base pairing rules, and how genetic information is encoded. Explain the process of DNA replication, including the roles of enzymes like DNA polymerase and helicase. Discuss how mutations can occur and their potential effects on protein function and organismal traits."
    ],
    200: [
        "In machine learning, explain the concept of the bias-variance tradeoff and its fundamental importance in model selection. How does model complexity relate to bias and variance, and what are the consequences of high bias or high variance? Discuss various regularization techniques including L1 (Lasso) and L2 (Ridge) regularization, and explain how they work to prevent overfitting. Additionally, describe ensemble methods like bagging and boosting, and how they can help achieve better bias-variance balance.",
        "Describe the complete process of cellular respiration in eukaryotic cells, including the four main stages: glycolysis, pyruvate oxidation, the Krebs cycle, and the electron transport chain. Explain where each stage occurs within the cell, what molecules are consumed and produced, and how ATP is generated through substrate-level phosphorylation and oxidative phosphorylation. Discuss the role of oxygen as the final electron acceptor and explain why cellular respiration is more efficient than fermentation in terms of ATP production per glucose molecule.",
        "Explain the concept of market equilibrium in microeconomics and how supply and demand interact to determine market price and quantity. Discuss what happens when markets are not in equilibrium, including the mechanisms that drive prices back to equilibrium. Explain the concept of price elasticity of demand and supply, including the factors that determine elasticity and how elasticity affects the responsiveness of quantity to price changes. Provide examples of elastic and inelastic goods and explain the implications for pricing strategies and tax policy.",
        "What is climate change, and what are its primary anthropogenic causes? Explain the greenhouse effect in detail, including how solar radiation interacts with Earth's atmosphere and surface. Discuss the role of carbon dioxide, methane, and other greenhouse gases in trapping heat, and explain how human activities such as burning fossil fuels, deforestation, and industrial processes have increased atmospheric concentrations of these gases. Describe the potential impacts of climate change on global temperature, precipitation patterns, sea levels, ecosystems, and human societies, and outline both mitigation and adaptation strategies.",
        "Describe the structure and function of DNA in comprehensive detail, including the double helix structure discovered by Watson and Crick, the complementary base pairing between adenine-thymine and guanine-cytosine, and how the sequence of bases encodes genetic information. Explain the process of DNA replication, including the roles of key enzymes like DNA polymerase, helicase, and ligase, and discuss the semi-conservative nature of replication. Explain how DNA mutations can occur through various mechanisms and discuss their potential effects on protein structure and function, as well as their role in evolution and disease."
    ],
    250: [
        "In machine learning, explain the fundamental concept of the bias-variance tradeoff and its critical importance in model selection and performance optimization. How does model complexity relate to bias and variance, and what are the specific consequences of high bias (underfitting) versus high variance (overfitting)? Discuss various regularization techniques in detail, including L1 (Lasso) regularization which promotes sparsity, L2 (Ridge) regularization which penalizes large weights, and elastic net which combines both approaches. Explain how these techniques work mathematically to prevent overfitting and improve generalization. Additionally, describe ensemble methods like bagging (Bootstrap Aggregating) and boosting, and explain how they can help achieve better bias-variance balance through different approaches to combining multiple models.",
        "Describe the complete process of cellular respiration in eukaryotic cells, covering all four main stages in detail: glycolysis in the cytoplasm, pyruvate oxidation in the mitochondrial matrix, the Krebs cycle (citric acid cycle) in the mitochondrial matrix, and the electron transport chain in the inner mitochondrial membrane. Explain the specific molecules consumed and produced at each stage, the role of key enzymes and cofactors, and how ATP is generated through both substrate-level phosphorylation and oxidative phosphorylation. Discuss the role of oxygen as the final electron acceptor in the electron transport chain and explain why cellular respiration is significantly more efficient than fermentation, producing approximately 30-32 ATP molecules per glucose molecule compared to only 2 ATP from fermentation.",
        "Explain the concept of market equilibrium in microeconomics and how the interaction of supply and demand determines both market price and quantity exchanged. Discuss what happens when markets are not in equilibrium, including the specific mechanisms that drive prices back to equilibrium through the actions of buyers and sellers. Explain the concept of price elasticity of demand and supply in detail, including the factors that determine whether demand or supply is elastic or inelastic, such as availability of substitutes, time horizon, and proportion of income spent on the good. Provide concrete examples of elastic and inelastic goods and services, and explain the implications of different elasticities for pricing strategies, tax policy, and market efficiency.",
        "What is climate change, and what are its primary anthropogenic causes? Explain the greenhouse effect in comprehensive detail, including how solar radiation interacts with Earth's atmosphere and surface, and how greenhouse gases trap infrared radiation that would otherwise escape to space. Discuss the specific roles of carbon dioxide, methane, nitrous oxide, and fluorinated gases in the greenhouse effect, and explain how human activities such as burning fossil fuels, deforestation, agriculture, and industrial processes have dramatically increased atmospheric concentrations of these gases since the Industrial Revolution. Describe the potential impacts of climate change on global temperature patterns, precipitation and weather systems, sea level rise, ocean acidification, ecosystem disruption, and human societies including agriculture, water resources, and public health. Outline both mitigation strategies to reduce greenhouse gas emissions and adaptation strategies to cope with unavoidable climate impacts.",
        "Describe the structure and function of DNA in comprehensive detail, including the double helix structure discovered by Watson and Crick, the complementary base pairing between adenine-thymine and guanine-cytosine, and how the specific sequence of nucleotide bases encodes genetic information for protein synthesis. Explain the process of DNA replication in detail, including the roles of key enzymes like DNA polymerase, helicase, primase, and ligase, and discuss the semi-conservative nature of replication where each new DNA molecule contains one original strand and one newly synthesized strand. Explain how DNA mutations can occur through various mechanisms including point mutations, insertions, deletions, and chromosomal rearrangements, and discuss their potential effects on protein structure and function, as well as their role in evolution, genetic diversity, and the development of genetic diseases."
    ],
    300: [
        "In machine learning, explain the fundamental concept of the bias-variance tradeoff and its critical importance in model selection and performance optimization. How does model complexity relate to bias and variance, and what are the specific consequences of high bias (underfitting) versus high variance (overfitting)? Discuss various regularization techniques in detail, including L1 (Lasso) regularization which promotes sparsity by driving coefficients to zero, L2 (Ridge) regularization which penalizes large weights to prevent overfitting, and elastic net which combines both approaches to balance sparsity and regularization. Explain how these techniques work mathematically to prevent overfitting and improve generalization performance. Additionally, describe ensemble methods like bagging (Bootstrap Aggregating) which reduces variance by averaging multiple models trained on different bootstrap samples, and boosting which reduces bias by sequentially training models to correct previous errors, and explain how they can help achieve better bias-variance balance through different approaches to combining multiple models.",
        "Describe the complete process of cellular respiration in eukaryotic cells, covering all four main stages in comprehensive detail: glycolysis occurring in the cytoplasm where glucose is broken down into pyruvate, pyruvate oxidation in the mitochondrial matrix where pyruvate is converted to acetyl-CoA, the Krebs cycle (citric acid cycle) in the mitochondrial matrix where acetyl-CoA is completely oxidized, and the electron transport chain in the inner mitochondrial membrane where the majority of ATP is produced. Explain the specific molecules consumed and produced at each stage, the role of key enzymes and cofactors like NAD+ and FAD, and how ATP is generated through both substrate-level phosphorylation during glycolysis and the Krebs cycle, and oxidative phosphorylation during the electron transport chain. Discuss the role of oxygen as the final electron acceptor in the electron transport chain and explain why cellular respiration is significantly more efficient than fermentation, producing approximately 30-32 ATP molecules per glucose molecule compared to only 2 ATP from fermentation, and how this efficiency advantage has driven the evolution of aerobic organisms."
    ],
    350: [
        "In machine learning, explain the fundamental concept of the bias-variance tradeoff and its critical importance in model selection and performance optimization. How does model complexity relate to bias and variance, and what are the specific consequences of high bias (underfitting) versus high variance (overfitting)? Discuss various regularization techniques in detail, including L1 (Lasso) regularization which promotes sparsity by driving coefficients to zero and is useful for feature selection, L2 (Ridge) regularization which penalizes large weights to prevent overfitting and is effective for multicollinearity, and elastic net which combines both approaches to balance sparsity and regularization. Explain how these techniques work mathematically through penalty terms added to the loss function, and how they prevent overfitting and improve generalization performance. Additionally, describe ensemble methods like bagging (Bootstrap Aggregating) which reduces variance by averaging multiple models trained on different bootstrap samples of the data, and boosting which reduces bias by sequentially training models to correct previous errors, and explain how they can help achieve better bias-variance balance through different approaches to combining multiple models and their respective advantages in different scenarios."
    ],
    400: [
        "In machine learning, explain the fundamental concept of the bias-variance tradeoff and its critical importance in model selection and performance optimization. How does model complexity relate to bias and variance, and what are the specific consequences of high bias (underfitting) versus high variance (overfitting)? Discuss various regularization techniques in detail, including L1 (Lasso) regularization which promotes sparsity by driving coefficients to zero and is particularly useful for feature selection in high-dimensional datasets, L2 (Ridge) regularization which penalizes large weights to prevent overfitting and is effective for handling multicollinearity in regression problems, and elastic net which combines both approaches to balance sparsity and regularization. Explain how these techniques work mathematically through penalty terms added to the loss function, and how they prevent overfitting and improve generalization performance by constraining model complexity. Additionally, describe ensemble methods like bagging (Bootstrap Aggregating) which reduces variance by averaging multiple models trained on different bootstrap samples of the data, and boosting which reduces bias by sequentially training models to correct previous errors, and explain how they can help achieve better bias-variance balance through different approaches to combining multiple models and their respective advantages in different scenarios and problem types."
    ]
}

def get_mmlu_prompt(seq_len: int, iteration: int = 0) -> str:
    """
    Get MMLU prompt for specified sequence length
    
    Args:
        seq_len: Target sequence length
        iteration: Iteration number for variety
        
    Returns:
        MMLU prompt text
    """
    if seq_len not in MMLU_PROMPTS:
        # Fallback to closest available length
        available_lengths = sorted(MMLU_PROMPTS.keys())
        closest_len = min(available_lengths, key=lambda x: abs(x - seq_len))
        seq_len = closest_len
    
    prompts = MMLU_PROMPTS[seq_len]
    return prompts[iteration % len(prompts)]

def tokenize_prompt(tokenizer, prompt: str, max_length: int) -> List[int]:
    """
    Tokenize MMLU prompt to target length
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt
        max_length: Target token length
        
    Returns:
        List of token IDs
    """
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    
    # Truncate or pad to target length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        # Pad with EOS token
        tokens.extend([tokenizer.eos_token_id] * (max_length - len(tokens)))
    
    return tokens
