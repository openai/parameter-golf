# Project Roadmap: Omnimodal AI Foundation

This document outlines the five-phase development roadmap for building the base foundational AI model.

The ultimate goal is a highly generalizable, safe, multimodal agent capable of complex reasoning, real-time interaction, computer agency, creative generation, and continual self-improvement.

## Development To-Do List

### Phase 1: Base Foundation & Core Training Strategy
**Goal:** Establish multimodal understanding across text, audio, and vision, with strict accuracy targets for high-priority domains.

**Priorities:** English (Text/Audio), Safety & Reliability, Math.

#### Dataset Curation & Architecture Prep
- [ ] Implement "Accuracy Architecture Debugging" mechanisms for monitoring learning signals.
- [ ] **Speech/Audio Collection:**
    - [ ] Real-time streaming support capability.
    - [ ] Multi-speaker support (up to 8 distinct speakers with blending capability).
    - [ ] Sounds with meaning (prosody, fillers like "hmm", "umm").
    - [ ] General sound effects library.
- [ ] **Knowledge Domain Collection:**
    - [ ] English: Comprehensive dataset ranging from very young age -> high school -> college level.
    - [ ] Math: Kindergarten -> PhD level dataset.
    - [ ] Sciences: Physics (College level), Biology (College level).
    - [ ] Machine Learning Education (College level).
    - [ ] Bible Training data (support for up to 120 languages).
    - [ ] Programming dataset (limited examples available initially).

#### Training & Accuracy Verification
*Targets based on custom architecture feasibility.*

- [ ] **High Priority Targets (Target: ≥95%):**
    - [ ] Train and verify English Text Understanding.
    - [ ] Train and verify English Audio Understanding.
    - [ ] Train and verify Math Understanding (K-PhD).
- [ ] **Medium Priority Targets:**
    - [ ] Train Image Understanding (Target: ≥95%).
    - [ ] Implement initial Safety Training protocols (Target: ~80% reliable initially).
    - [ ] Train Programming Accuracy (Target: ~80% due to dataset constraints).
- [ ] **Lower Priority Targets:**
    - [ ] Audio-Video Sync Training.
    - [ ] Train Video Understanding (Target: ≥70% acceptable).
    - [ ] Train Multiple Language Understanding (Target: ≥50% acceptable as lowest priority).

---

### Phase 2: Computer Agency & General Navigation
**Goal:** The model achieves generalized agency to navigate the web, utilize any operating system or application, and play video games.

**Note:** Safety and reliability are paramount in this phase. Utilizing custom architecture to achieve high learning efficiency (2-3 epochs).

- [ ] **Generalization Training:**
    - [ ] Train model on generalized OS interaction (Windows, Linux, macOS without specific pre-programming).
    - [ ] Train generalized App usage capability.
- [ ] **Web & Gaming:**
    - [ ] Train autonomous web navigation and search capabilities.
    - [ ] Train generalized video game playing capability.
- [ ] **Safety & Accuracy Targets:**
    - [ ] Verify Computer Usage/Navigation accuracy (Target: **≥98% accurate**).
    - [ ] Implement rigorous safety checks for autonomous actions.

---

### Phase 3: Creation, Personality & Advanced Programming
**Goal:** Evolve the model into a creative engine capable of 3D generation, acting as a game engine simulator, refining its programming skills, and developing a consistent persona.

#### The Creative Engine & Simulation
- [ ] **3D Asset Generation:**
    - [ ] Train generation of 3D models.
    - [ ] Implement automatic skinning and weight painting capabilities.
    - [ ] Implement texture generation.
    - [ ] Generate expressive face shapes for accurate talking animations.
- [ ] **Game Engine Simulation:**
    - [ ] Train physics understanding for simulation.
    - [ ] Implement 3D world generation with tiling.
    - [ ] Train NPC control and dialogue generation.
    - [ ] **Verification:** Ensure high accuracy in simulation styling, realism/fantasy interactions, and social creature interactions.
    - [ ] Implement consistent game input triggering within the world simulation.

#### Personality & Tools
- [ ] **Personality Consistency:** Implement a consistent persona that influences speech while retaining all Phase 1 & 2 knowledge.
- [ ] **Advanced Programming:**
    - [ ] Rigorous re-training on programming datasets for higher accuracy.
    - [ ] Enable capability to connect to and use MCP (Model Context Protocol) servers.
    - [ ] Enable autonomous tool creation capability.

---

### Phase 4: VR Interactions
**Goal:** Extend the model's agency into virtual reality environments.

- [ ] **Integration:** Develop VRChat integration protocols.
- [ ] **Interaction:** Train the model to control 3D avatars for interactions with humans and other agents in VR space.
- [ ] **Compatible:** Compatible with VR Chat (Unity) and Warudo (VTuber) software.

---

### Phase 5: UI, UX, and Lifecycle Management
*(Self-Correction Note: As suggested in planning, elements of this phase regarding knowledge merging might be prerequisites for successfully completing Phase 3 without data loss).*

**Goal:** Ensure usability, manageability, and robust continual learning capabilities.

- [ ] **User Experience (UX):**
    - [ ] Develop clear documentation for usage.
    - [ ] Implement clear options for Voice selection.
    - [ ] Implement clear options for Model selection.
    - [ ] Create granular access control options (computer permissions/tools).
- [ ] **Lifecycle & Continual Learning:**
    - [ ] **Crucial:** Develop mechanisms to merge new knowledge with old knowledge without catastrophic forgetting (losing prior knowledge). This is possible with json recipes and seeds (with the saved torch seed too) all being stored in readable json. Maybe merge version can have an auto check for collsions (different memories saved to the same recipe spots) and maybe it can also check to be sure before merging that both SHA seeds are the same and if there is a difference, then it may not be compatible? 
    - [ ] Enable autonomous self-learning and training scheduling.
    - [ ] Implement capabilities for the model to manage subordinate AI agents.

### Phase 6: UI, UX, and Lifecycle Management
*(Self-Correction Note: As suggested in planning, elements of this phase regarding knowledge merging might be prerequisites for successfully completing Phase 3 without data loss).*

**Goal:** Ensure usability, manageability, and robust continual learning capabilities.

- [ ] **User Experience (UX):**
    - [ ] Develop clear documentation for usage.
    - [ ] Implement clear options for Voice selection.
    - [ ] Implement clear options for Model selection.
    - [ ] Create granular access control options (computer permissions/tools).
- [ ] **Lifecycle & Continual Learning:**
    - [ ] **Crucial:** Develop mechanisms to merge new knowledge with old knowledge without catastrophic forgetting (losing prior knowledge). This is possible with json recipes and seeds (with the saved torch seed too) all being stored in readable json. Maybe merge version can have an auto check for collsions (different memories saved to the same recipe spots) and maybe it can also check to be sure before merging that both SHA seeds are the same and if there is a difference, then it may not be compatible? 
    - [ ] Enable autonomous self-learning and training scheduling.
    - [ ] Implement capabilities for the model to manage subordinate AI agents.