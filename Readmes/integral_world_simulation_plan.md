# Plan for Integral AI-Style World Simulation in HDC_Sparse

This document outlines a comprehensive plan to upgrade the HDC_Sparse model to incorporate an "Integral AI-style" world simulation. This involves moving from learning *from* a simulator to learning *to be* a simulator, enabling the model to build an internal predictive model of physics, explore autonomously, and simulate future scenarios for safety and planning.

## 1. Internalize the Physics Engine (HDC World Model)

**Goal:** Enable the HDC system to predict the outcome of actions without relying on the external Mujoco simulator for every step. This creates a "neural physics engine" within the HDC memory.

### 1.1. New Relationship Type: `PREDICTS`
*   **Concept:** Introduce a new relationship type `PREDICTS` to the `RelationshipType` enum and `RelationshipEncoder`.
*   **Mechanism:** Bind `(current_state, action)` to `predicted_next_state`.
    *   `prediction_vec = bind(bind(current_state_vec, action_vec), PREDICTS_marker)`
    *   `memory.store(prediction_vec, predicted_next_state_vec)`
*   **Implementation:**
    *   Update `Hdc_Sparse/relationship_encoder.py` to include `PREDICTS`.
    *   Update `Hdc_Sparse/hdc_sparse_core.py` or a new module `Hdc_Sparse/predictive_model.py` to handle state-action-prediction binding.

### 1.2. State and Action Encoding
*   **State Encoding:** Leverage existing `SevenSenseWorldAdapter3D` to create a comprehensive state vector `S_t` including:
    *   Proprioception (joint angles, velocities)
    *   Vestibular (head orientation, acceleration)
    *   Visual/Spatial (object positions, occupancy)
    *   Haptic/Texture (contact forces)
*   **Action Encoding:** Define a standard set of actions (e.g., `apply_force`, `move_joint`) encoded as HDC vectors `A_t`.

### 1.3. Predictive Loop (Mental Time Travel)
*   **Simulation:** Implement a function `simulate_trajectory(initial_state, action_sequence)` that runs entirely in HDC space.
    *   `S_0 = initial_state`
    *   For `A_t` in `action_sequence`:
        *   `S_{t+1} = memory.query(bind(S_t, A_t, PREDICTS))`
*   **Benefit:** Allows rapid evaluation of potential plans without expensive physics stepping.

## 2. Switch from "Curriculum" to "Curiosity" (Self-Supervised Exploration)

**Goal:** Replace static datasets with an autonomous exploration loop where the agent generates its own training data by interacting with the world.

### 2.1. Curiosity-Driven Exploration Loop
*   **Mechanism:**
    1.  **Observe** current state `S_t`.
    2.  **Propose** candidate actions `{A_1, A_2, ...}`.
    3.  **Predict** outcomes `{P_1, P_2, ...}` using the internal HDC model.
    4.  **Select** action `A_k` that maximizes *uncertainty* or *prediction error* (initially) or *expected reward* (later).
        *   Uncertainty can be measured by the "confidence" or density of the retrieved prediction vector.
    5.  **Act** in the real Mujoco simulator: `S_{t+1_real} = env.step(A_k)`.
    6.  **Learn:**
        *   Compute error: `E = similarity(P_k, S_{t+1_real})`.
        *   Update HDC memory: Store `bind(S_t, A_k) -> S_{t+1_real}`.
        *   If error was high, this is a high-value learning event.

### 2.2. Zero-Knowledge Spawning
*   **Setup:** Spawn the agent in `MujocoWorld3D` with no pre-loaded recipes.
*   **Babbling:** Initially, the agent performs random motor babbling to learn basic `(state, action) -> next_state` primitives.
*   **Mastery:** As the internal model improves, shift from maximizing error (curiosity) to minimizing error (mastery/planning).

## 3. Enhanced Sleep as "Counterfactual Simulation" (Dreaming of the Future)

**Goal:** Use sleep cycles not just to consolidate past memories, but to simulate future scenarios, particularly dangerous ones, to learn safety constraints without real-world risks.

### 3.1. "Dreaming of the Future" Mode
*   **Trigger:** During sleep consolidation phases (already present in `sleep_consolidation.py`).
*   **Process:**
    1.  **Sample** a starting state `S_seed` from recent experiences.
    2.  **Generate** hypothetical action sequences (e.g., "walk off cliff", "touch hot surface").
    3.  **Simulate** outcomes using the `PREDICTS` relationships learned in Step 1.
    4.  **Evaluate:** Check if the predicted state `S_final` violates safety constraints (e.g., high impact force, extreme temperature).
*   **Synthetic Experience:** If a dangerous outcome is predicted, store a "negative" or "avoidance" rule: `bind(S_seed, action) -> DANGER`.

### 3.2. Safety Constraints
*   **Definition:** Define safety predicates based on sensory inputs (e.g., `vestibular.acceleration > threshold` = crash).
*   **Integration:** The `ExtendedReasoningSystem` (from v2.4.3) can be used to manage these safety goals and constraints.

## 4. Creative Synthesis (Inventing New Primitives)

**Goal:** Enhance `HDC Program Synthesis` to invent entirely new low-level functions rather than just composing existing ones.

### 4.1. Primitive Discovery
*   **Observation:** Identify repeated sequences of atomic actions that lead to useful state changes.
*   **Chunking:** Bundle these sequences into a new atomic primitive `P_new`.
    *   `P_new = bundle(sequence(a1, a2, a3))`
*   **Re-use:** Add `P_new` to the available action set for the Curiosity Loop (Step 2).
*   **Example:** Discovering "jump" by combining "crouch" + "extend_legs" + "push".

## 5. Implementation Roadmap

### Phase 1: Foundation (The Predictor)
*   [ ] Modify `RelationshipType` to include `PREDICTS`.
*   [ ] Create `PredictiveModel` class wrapping `SevenSenseSparseMemory`.
*   [ ] Implement `encode_state` and `encode_action` functions bridging `MujocoWorld3D` and HDC.

### Phase 2: The Explorer (Curiosity Loop)
*   [ ] Create `CuriosityAgent` class.
*   [ ] Implement the `observe -> predict -> act -> learn` loop.
*   [ ] Connect to `MujocoWorld3D` for real-time stepping.

### Phase 3: The Dreamer (Counterfactuals)
*   [ ] Extend `SleepConsolidation` to support "future simulation".
*   [ ] Implement safety checks on predicted states.
*   [ ] Store avoidance rules.

### Phase 4: Integration
*   [ ] Integrate with `train_arc_agi2.py` or create a new `train_world_model.py`.
*   [ ] Validate on simple physics tasks (e.g., object pushing, stability).

## 6. File Structure Changes

*   `Hdc_Sparse/predictive_model.py`: New module for state-action prediction.
*   `Hdc_Sparse/curiosity_agent.py`: New module for the exploration loop.
*   `Hdc_Sparse/world_model_training.py`: Main training script for this new mode.
*   `Game_APP_ARC_AGI_3_training/physics_world_3d.py`: (Existing) Source of truth for physics.

This plan moves HDC_Sparse significantly closer to the "Integral AI" vision of an innate, self-supervised world simulator.

## 7
Improvement: Enhance the HDC Program Synthesis (v2.5.0) to invent new primitives. Currently, it composes existing ones (Rotate + Flip). A true AGI should be able to write entirely new low-level functions (e.g., discovering a "pixel-flood-fill" algorithm from scratch) rather than just using pre-built blocks.

## 8 
Video game learning from distillation (a teacher model) such as this one: https://huggingface.co/nvidia/NitroGen (This model is only 1 Billion parameters so it won't need an elaborate setup)

Web UI usage distillation learning: UI-Venus - git clone https://huggingface.co/inclusionAI/UI-Venus-Ground-7B (Need to use this as a ground distillation model for the framework below to teach the HDC Binding model (student) for how to use the GUI and OS.) (This requires a more elaborate setup and probably either one powerful GPU or a cluster of two less powerful GPUs for teaching)

Web UI and Browser Cross learning (distillation Teacher): https://github.com/simular-ai/Agent-S.git

## 9 

Other various domains of knowledge:

Historical (smithsonian): https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/index.txt

ArXiv (filtered for CC0/CC-By): https://arxiv.org/

Programming(OpenStacks2): https://huggingface.co/datasets/bigcode/the-stack-v2-dedup

Wikipedia Official huggingface datasets (to abide by their rules): wikimedia/wikipedia (already present in the model training file)

Geology textbook: https://opengeology.org/textbook/

Open images v7 (huggingface): bitmind/open-images-v7 (not already included)
The gold standard for this is OpenStax.

Video Understanding: nkp37/OpenVid-1M (huggingface - Not already included in training)

OpenStax (Best for General Knowledge)
OpenStax is a nonprofit based at Rice University. They provide peer-reviewed textbooks that are equivalent to the big commercial publishers but are licensed under Creative Commons Attribution 4.0 (CC BY). This is a huggingface download: crumb/openstax-text. (Not included in the training data yet.)

Multilingual understanding and translation for over 100 languages with the Bible Corpus: https://textgridrep.org/project/TGPR-d862e14d-4df7-052b-00fe-661cb242231c#README