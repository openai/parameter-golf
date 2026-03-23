"""
HDC Template Library with Compositional Dynamic Creation

Templates are reusable transformation patterns with slot variables.
They enable:
- Pattern matching for task decomposition
- Dynamic instantiation with specific values
- Compositional creation of complex transformations
- Tiered complexity learning (ATOMIC → COMPOUND → CONDITIONAL → RECURSIVE)

Key Concepts:
- Templates: Recipes with unbound variable slots
- Slots: Placeholders that get bound at instantiation time
- Prototypes: Pre-computed HDC vectors for fast template matching
- Composition: Building complex templates from simpler ones

Example:
    >>> template = Template("rotate", ["INPUT", "ANGLE"])
    >>> result = template.instantiate({"INPUT": grid_vec, "ANGLE": angle_90_vec})
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from ..HDC_Core_Main.hdc_sparse_core import SparseBinaryHDC, AtomicVocabulary


def _string_to_seed(s: str) -> int:
    """Deterministic seed generation for reproducible HDC vectors."""
    hash_bytes = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF


def bind_all(vectors: list, hdc=None) -> np.ndarray:
    """
    Combine multiple vectors using XOR binding (NOT bundling).
    100% lossless per README.
    """
    if not vectors:
        if hdc is not None:
            return hdc.zeros()
        return np.array([], dtype=np.uint8)
    
    result = np.zeros_like(vectors[0])
    for i, vec in enumerate(vectors):
        pos_seed = _string_to_seed(f"BIND_ALL_POS::{i}")
        if hdc is not None:
            pos_vec = hdc.from_seed(pos_seed)
        else:
            # Use SHA256-based deterministic generation for cross-platform reproducibility
            pos_vec = _sha256_deterministic_bytes(pos_seed, len(vec))
        positioned = np.bitwise_xor(vec, pos_vec)
        result = np.bitwise_xor(result, positioned)
    return result


def _sha256_deterministic_bytes(seed: int, num_bytes: int) -> np.ndarray:
    """
    Generate deterministic bytes using SHA256 hashing.
    
    This method is 100% deterministic across:
    - All hardware platforms (x86, ARM, GPU, etc.)
    - All programming languages (Python, C++, Rust, etc.)
    - All operating systems (Windows, Linux, macOS, etc.)
    - All versions of libraries (no NumPy version dependency)
    
    Args:
        seed: Integer seed for deterministic generation
        num_bytes: Number of bytes to generate
        
    Returns:
        NumPy array of uint8 bytes
    """
    result = b''
    counter = 0
    
    while len(result) < num_bytes:
        # Create deterministic input: seed + counter
        data = f"{seed}:{counter}".encode('utf-8')
        hash_bytes = hashlib.sha256(data).digest()
        result += hash_bytes
        counter += 1
    
    return np.frombuffer(result[:num_bytes], dtype=np.uint8).copy()


class TemplateComplexity(Enum):
    """
    Tiered complexity levels for templates.
    
    Based on cognitive complexity:
    - ATOMIC: Single operation (rotate, flip)
    - COMPOUND: Two operations composed
    - CONDITIONAL: IF-THEN rules
    - RECURSIVE: Iterative/self-referential rules
    - ABSTRACT: Analogy-based patterns
    """
    ATOMIC = 1       # Single op: rotate(INPUT, ANGLE)
    COMPOUND = 2     # Two ops: compose(rotate, flip)
    CONDITIONAL = 3  # IF-THEN: if(CONDITION, THEN, ELSE)
    RECURSIVE = 4    # Apply until: repeat(OP, UNTIL)
    ABSTRACT = 5     # Analogy: as(A_TO_B, C_TO_?)


@dataclass
class TemplateSlot:
    """
    A slot/variable in a template.
    
    Slots are placeholders that get bound to specific values
    when the template is instantiated.
    """
    name: str                          # Slot name (e.g., "INPUT", "ANGLE")
    role: str = "input"                # Role: input, modifier, output
    required: bool = True              # Is binding required?
    default_seed: Optional[int] = None # Default value seed if not bound
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'role': self.role,
            'required': self.required,
            'default_seed': self.default_seed
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TemplateSlot':
        return cls(**d)


@dataclass 
class Template:
    """
    A template with named slots for compositional pattern creation.
    
    Templates are "recipes with holes" - the slots are the holes
    that get filled in when the template is instantiated.
    
    Example:
        rotate(INPUT, ANGLE) where INPUT and ANGLE are slots
        → instantiate with INPUT=grid_vec, ANGLE=90_vec
        → returns bind(bind(grid_vec, 90_vec), rotate_prototype)
    """
    name: str
    slots: List[TemplateSlot]
    complexity: TemplateComplexity = TemplateComplexity.ATOMIC
    description: str = ""
    prototype_seed: Optional[int] = None
    
    # Operation chain (how to combine slots)
    operations: List[Dict] = field(default_factory=list)
    
    # For compound templates: sub-template references
    sub_templates: List[str] = field(default_factory=list)
    
    def slot_names(self) -> List[str]:
        """Get list of slot names."""
        return [s.name for s in self.slots]
    
    def required_slots(self) -> List[str]:
        """Get names of required slots."""
        return [s.name for s in self.slots if s.required]
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'slots': [s.to_dict() for s in self.slots],
            'complexity': self.complexity.value,
            'description': self.description,
            'prototype_seed': self.prototype_seed,
            'operations': self.operations,
            'sub_templates': self.sub_templates
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Template':
        slots = [TemplateSlot.from_dict(s) for s in d['slots']]
        return cls(
            name=d['name'],
            slots=slots,
            complexity=TemplateComplexity(d['complexity']),
            description=d.get('description', ''),
            prototype_seed=d.get('prototype_seed'),
            operations=d.get('operations', []),
            sub_templates=d.get('sub_templates', [])
        )


class TemplateLibrary:
    """
    Library of reusable HDC templates.
    
    Templates are recipes with slot variables that enable:
    1. Pattern matching: Find which template best matches a task
    2. Instantiation: Fill slots with specific values
    3. Composition: Build complex templates from simpler ones
    
    Pre-built categories:
    - GEOMETRIC: rotate, flip, scale, translate
    - COLOR: color_swap, color_fill, color_map
    - STRUCTURAL: extract, insert, tile, crop
    - COMPOSITIONAL: compose_two, apply_sequence, conditional
    
    Example:
        >>> lib = TemplateLibrary(hdc)
        >>> matches = lib.match_template(task_vec, top_k=3)
        >>> result = lib.instantiate("rotate", {"INPUT": grid, "ANGLE": 90})
    """
    
    def __init__(self, hdc: SparseBinaryHDC):
        """Initialize template library."""
        self.hdc = hdc
        self.templates: Dict[str, Template] = {}
        self.prototypes: Dict[str, np.ndarray] = {}  # Template name → prototype vector
        
        # Build default templates
        self._build_default_templates()
    
    def _string_to_seed(self, s: str) -> int:
        """Convert string to deterministic seed."""
        h = hashlib.sha256(s.encode()).digest()
        return int.from_bytes(h[:8], 'big')
    
    def _get_prototype(self, template_name: str) -> np.ndarray:
        """Get or create prototype vector for template."""
        if template_name not in self.prototypes:
            seed = self._string_to_seed(f"template_prototype_{template_name}")
            self.prototypes[template_name] = self.hdc.from_seed(seed)
        return self.prototypes[template_name]
    
    def _build_default_templates(self):
        """Build the default template library."""
        
        # =====================================================================
        # GEOMETRIC TEMPLATES (ATOMIC)
        # =====================================================================
        
        self.add_template(Template(
            name="rotate",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("ANGLE", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Rotate input by angle (90, 180, 270)",
            operations=[
                {"op": "bind", "args": ["INPUT", "ANGLE"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="flip",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("DIRECTION", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Flip input in direction (horizontal, vertical, diagonal)",
            operations=[
                {"op": "bind", "args": ["INPUT", "DIRECTION"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="scale",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("FACTOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Scale input by factor (2x, half, etc.)",
            operations=[
                {"op": "bind", "args": ["INPUT", "FACTOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="translate",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("DIRECTION", role="modifier"),
                TemplateSlot("DISTANCE", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Move input in direction by distance",
            operations=[
                {"op": "bind", "args": ["INPUT", "DIRECTION"]},
                {"op": "bind", "args": ["_result", "DISTANCE"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        # =====================================================================
        # COLOR TEMPLATES (ATOMIC)
        # =====================================================================
        
        self.add_template(Template(
            name="color_swap",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("COLOR1", role="modifier"),
                TemplateSlot("COLOR2", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Swap two colors in input",
            operations=[
                {"op": "bind", "args": ["INPUT", "COLOR1"]},
                {"op": "bind", "args": ["_result", "COLOR2"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_fill",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("REGION", role="modifier"),
                TemplateSlot("COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Fill region with color",
            operations=[
                {"op": "bind", "args": ["INPUT", "REGION"]},
                {"op": "bind", "args": ["_result", "COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_map",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("MAPPING", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Apply color mapping transformation",
            operations=[
                {"op": "bind", "args": ["INPUT", "MAPPING"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_replace",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("OLD_COLOR", role="modifier"),
                TemplateSlot("NEW_COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Replace all occurrences of old color with new color",
            operations=[
                {"op": "bind", "args": ["INPUT", "OLD_COLOR"]},
                {"op": "bind", "args": ["_result", "NEW_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_invert",
            slots=[
                TemplateSlot("INPUT", role="input")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Invert all colors (background becomes foreground pattern)",
            operations=[
                {"op": "invert", "args": ["INPUT"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_most_common",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("TARGET_COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Fill with the most common color or target color",
            operations=[
                {"op": "bind", "args": ["INPUT", "TARGET_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_by_position",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("POSITION", role="modifier"),
                TemplateSlot("COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Color specific positions with a color",
            operations=[
                {"op": "bind", "args": ["INPUT", "POSITION"]},
                {"op": "bind", "args": ["_result", "COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_count",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Count occurrences of a specific color",
            operations=[
                {"op": "unbind", "args": ["INPUT", "COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_highlight",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("COLOR", role="modifier"),
                TemplateSlot("HIGHLIGHT_COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Highlight cells of specific color with highlight color",
            operations=[
                {"op": "bind", "args": ["INPUT", "COLOR"]},
                {"op": "bind", "args": ["_result", "HIGHLIGHT_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="color_boundary",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("BOUNDARY_COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Color the boundary/edge cells",
            operations=[
                {"op": "bind", "args": ["INPUT", "BOUNDARY_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        # =====================================================================
        # GRAVITY TEMPLATES (ATOMIC) - Objects falling/moving in directions
        # =====================================================================
        
        self.add_template(Template(
            name="gravity_down",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("OBJECT_COLOR", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Apply gravity downward - objects fall to bottom",
            operations=[
                {"op": "bind", "args": ["INPUT", "OBJECT_COLOR"]},
                {"op": "permute", "args": ["_result", 64]},  # Encode direction
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="gravity_up",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("OBJECT_COLOR", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Apply gravity upward - objects rise to top",
            operations=[
                {"op": "bind", "args": ["INPUT", "OBJECT_COLOR"]},
                {"op": "permute", "args": ["_result", 128]},  # Different direction encoding
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="gravity_left",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("OBJECT_COLOR", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Apply gravity leftward - objects move to left edge",
            operations=[
                {"op": "bind", "args": ["INPUT", "OBJECT_COLOR"]},
                {"op": "permute", "args": ["_result", 192]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="gravity_right",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("OBJECT_COLOR", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Apply gravity rightward - objects move to right edge",
            operations=[
                {"op": "bind", "args": ["INPUT", "OBJECT_COLOR"]},
                {"op": "permute", "args": ["_result", 256]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="gravity_diagonal",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("DIRECTION", role="modifier"),
                TemplateSlot("OBJECT_COLOR", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Apply diagonal gravity (SE, SW, NE, NW)",
            operations=[
                {"op": "bind", "args": ["INPUT", "DIRECTION"]},
                {"op": "bind", "args": ["_result", "OBJECT_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="gravity_stack",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("DIRECTION", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Stack objects in a direction (compact/pack)",
            operations=[
                {"op": "bind", "args": ["INPUT", "DIRECTION"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="gravity_until_obstacle",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("DIRECTION", role="modifier"),
                TemplateSlot("OBSTACLE_COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.COMPOUND,
            description="Objects move until hitting obstacle of specific color",
            operations=[
                {"op": "bind", "args": ["INPUT", "DIRECTION"]},
                {"op": "bind", "args": ["_result", "OBSTACLE_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="gravity_bounce",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("DIRECTION", role="modifier"),
                TemplateSlot("BOUNCE_COLOR", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.COMPOUND,
            description="Objects move and bounce off edges/obstacles",
            operations=[
                {"op": "bind", "args": ["INPUT", "DIRECTION"]},
                {"op": "bind", "args": ["_result", "BOUNCE_COLOR"]},
                {"op": "permute", "args": ["_result", 320]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        # =====================================================================
        # STRUCTURAL TEMPLATES (ATOMIC)
        # =====================================================================
        
        self.add_template(Template(
            name="extract",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("PATTERN", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Extract matching pattern from input",
            operations=[
                {"op": "unbind", "args": ["INPUT", "PATTERN"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        # =====================================================================
        # MORPHOLOGICAL OUTLINE TEMPLATES (for MORPH_OUTLINE tasks)
        # =====================================================================
        
        self.add_template(Template(
            name="mark_boundary",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("BOUNDARY_COLOR", role="modifier"),
                TemplateSlot("INTERIOR_COLOR", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Mark boundary cells of a filled shape with boundary_color",
            operations=[
                {"op": "bind", "args": ["INPUT", "BOUNDARY_COLOR"]},
                {"op": "bind", "args": ["_result", "INTERIOR_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="mark_boundary_8conn",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("BOUNDARY_COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Mark boundary using 8-connectivity (includes diagonals)",
            operations=[
                {"op": "bind", "args": ["INPUT", "BOUNDARY_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="extract_boundary",
            slots=[
                TemplateSlot("INPUT", role="input")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Extract only boundary cells, remove interior (hollow outline)",
            operations=[
                {"op": "bind", "args": ["INPUT", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="extract_interior",
            slots=[
                TemplateSlot("INPUT", role="input")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Extract only interior cells, remove boundary",
            operations=[
                {"op": "invert", "args": ["INPUT"]},  # Conceptually opposite of boundary
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="dilate",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("ITERATIONS", role="modifier", required=False),
                TemplateSlot("FILL_COLOR", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Dilate/grow shape by expanding into adjacent background",
            operations=[
                {"op": "bind", "args": ["INPUT", "ITERATIONS"]},
                {"op": "bind", "args": ["_result", "FILL_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="erode",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("ITERATIONS", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Erode/shrink shape by removing boundary cells",
            operations=[
                {"op": "bind", "args": ["INPUT", "ITERATIONS"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="morph_outline_recolor",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("BOUNDARY_COLOR", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Recolor boundary cells of shape (MORPH_OUTLINE specific)",
            operations=[
                {"op": "bind", "args": ["INPUT", "BOUNDARY_COLOR"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="detect_and_mark_boundary",
            slots=[
                TemplateSlot("INPUT", role="input")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Auto-detect shape and mark its boundary with a new color",
            operations=[
                {"op": "bind", "args": ["INPUT", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="insert",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("OBJECT", role="modifier"),
                TemplateSlot("POSITION", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Insert object at position in input",
            operations=[
                {"op": "bind", "args": ["OBJECT", "POSITION"]},
                {"op": "bundle", "args": ["INPUT", "_result"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="tile",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("REPETITIONS", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Tile/repeat input pattern",
            operations=[
                {"op": "bind", "args": ["INPUT", "REPETITIONS"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="crop",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("REGION", role="modifier")
            ],
            complexity=TemplateComplexity.ATOMIC,
            description="Crop input to region",
            operations=[
                {"op": "bind", "args": ["INPUT", "REGION"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        # =====================================================================
        # COMPOUND TEMPLATES
        # =====================================================================
        
        self.add_template(Template(
            name="compose_two",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("T1", role="modifier"),
                TemplateSlot("T2", role="modifier")
            ],
            complexity=TemplateComplexity.COMPOUND,
            description="Apply two transformations in sequence: T2(T1(INPUT))",
            sub_templates=["_T1", "_T2"],
            operations=[
                {"op": "bind", "args": ["INPUT", "T1"]},
                {"op": "bind", "args": ["_result", "T2"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="apply_sequence",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("SEQUENCE", role="modifier")
            ],
            complexity=TemplateComplexity.COMPOUND,
            description="Apply a sequence of transformations",
            operations=[
                {"op": "bind", "args": ["INPUT", "SEQUENCE"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="parallel",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("T1", role="modifier"),
                TemplateSlot("T2", role="modifier")
            ],
            complexity=TemplateComplexity.COMPOUND,
            description="Apply two transformations in parallel and bundle",
            operations=[
                {"op": "bind", "args": ["INPUT", "T1"], "save_as": "r1"},
                {"op": "bind", "args": ["INPUT", "T2"], "save_as": "r2"},
                {"op": "bundle", "args": ["r1", "r2"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        # =====================================================================
        # CONDITIONAL TEMPLATES
        # =====================================================================
        
        self.add_template(Template(
            name="conditional",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("CONDITION", role="modifier"),
                TemplateSlot("THEN", role="modifier"),
                TemplateSlot("ELSE", role="modifier", required=False)
            ],
            complexity=TemplateComplexity.CONDITIONAL,
            description="IF condition THEN apply transform ELSE other",
            operations=[
                {"op": "bind", "args": ["CONDITION", "THEN"]},
                {"op": "bind", "args": ["_result", "INPUT"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="when_match",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("PATTERN", role="modifier"),
                TemplateSlot("ACTION", role="modifier")
            ],
            complexity=TemplateComplexity.CONDITIONAL,
            description="When input matches pattern, apply action",
            operations=[
                {"op": "unbind", "args": ["INPUT", "PATTERN"]},
                {"op": "bind", "args": ["_result", "ACTION"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        # =====================================================================
        # RECURSIVE TEMPLATES
        # =====================================================================
        
        self.add_template(Template(
            name="repeat_until",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("OPERATION", role="modifier"),
                TemplateSlot("TERMINATION", role="modifier")
            ],
            complexity=TemplateComplexity.RECURSIVE,
            description="Repeat operation until termination condition",
            operations=[
                {"op": "bind", "args": ["INPUT", "OPERATION"]},
                {"op": "bind", "args": ["_result", "TERMINATION"]},
                {"op": "permute", "args": ["_result", 1]},  # Encode recursion
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        self.add_template(Template(
            name="apply_n_times",
            slots=[
                TemplateSlot("INPUT", role="input"),
                TemplateSlot("OPERATION", role="modifier"),
                TemplateSlot("COUNT", role="modifier")
            ],
            complexity=TemplateComplexity.RECURSIVE,
            description="Apply operation N times",
            operations=[
                {"op": "bind", "args": ["INPUT", "OPERATION"]},
                {"op": "bind", "args": ["_result", "COUNT"]},
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
        
        # =====================================================================
        # ABSTRACT TEMPLATES
        # =====================================================================
        
        self.add_template(Template(
            name="analogy",
            slots=[
                TemplateSlot("A", role="input"),
                TemplateSlot("B", role="input"),
                TemplateSlot("C", role="input")
            ],
            complexity=TemplateComplexity.ABSTRACT,
            description="A:B :: C:? - Find what relates to C as B relates to A",
            operations=[
                {"op": "unbind", "args": ["B", "A"]},  # Get A→B relation
                {"op": "bind", "args": ["C", "_result"]},  # Apply to C
                {"op": "bind", "args": ["_result", "_prototype"]}
            ]
        ))
    
    # =========================================================================
    # Template Management
    # =========================================================================
    
    def add_template(self, template: Template):
        """Add a template to the library."""
        self.templates[template.name] = template
        # Create prototype
        seed = template.prototype_seed or self._string_to_seed(f"template_{template.name}")
        template.prototype_seed = seed
        self.prototypes[template.name] = self.hdc.from_seed(seed)
    
    def get_template(self, name: str) -> Optional[Template]:
        """Get template by name."""
        return self.templates.get(name)
    
    def list_templates(
        self,
        complexity: Optional[TemplateComplexity] = None
    ) -> List[str]:
        """List template names, optionally filtered by complexity."""
        if complexity is None:
            return list(self.templates.keys())
        return [
            name for name, t in self.templates.items()
            if t.complexity == complexity
        ]
    
    # =========================================================================
    # Template Matching
    # =========================================================================
    
    def match_template(
        self,
        task_vec: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
        max_complexity: Optional[TemplateComplexity] = None
    ) -> List[Tuple[str, float]]:
        """
        Find templates that best match a task vector.
        
        GPU-OPTIMIZED: Uses batch_similarity() for O(1) comparison
        against all templates instead of O(N) sequential calls.
        
        Args:
            task_vec: HDC vector representing the task
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold
            max_complexity: Maximum complexity to consider
        
        Returns:
            List of (template_name, similarity) tuples, sorted descending
        """
        # Filter templates by complexity first
        filtered_names = []
        filtered_protos = []
        
        for name, prototype in self.prototypes.items():
            template = self.templates.get(name)
            if template is None:
                continue
            if max_complexity and template.complexity.value > max_complexity.value:
                continue
            filtered_names.append(name)
            filtered_protos.append(prototype)
        
        if not filtered_protos:
            return []
        
        # GPU-OPTIMIZED: Batch similarity computation
        if hasattr(self.hdc, 'batch_similarity') and len(filtered_protos) > 1:
            # Single GPU kernel call for ALL templates
            # batch_similarity(query, candidates) returns similarities of query against all candidates
            similarities = self.hdc.batch_similarity(task_vec, filtered_protos)
            
            # Create matches list
            matches = [
                (name, float(sim)) 
                for name, sim in zip(filtered_names, similarities)
                if sim >= min_similarity
            ]
        else:
            # Fallback to sequential
            matches = []
            for name, prototype in zip(filtered_names, filtered_protos):
                sim = self.hdc.similarity(task_vec, prototype)
                if sim >= min_similarity:
                    matches.append((name, float(sim)))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
    
    def match_complexity_first(
        self,
        task_vec: np.ndarray,
        min_similarity: float = 0.5
    ) -> Optional[str]:
        """
        Find simplest template that matches.
        
        Implements Occam's razor: prefer simpler explanations.
        Tries ATOMIC first, then COMPOUND, etc.
        
        Args:
            task_vec: Task vector to match
            min_similarity: Minimum similarity threshold
        
        Returns:
            Template name or None if no match
        """
        for complexity in TemplateComplexity:
            matches = self.match_template(
                task_vec,
                top_k=1,
                min_similarity=min_similarity,
                max_complexity=complexity
            )
            if matches:
                return matches[0][0]
        return None
    
    # =========================================================================
    # Template Instantiation
    # =========================================================================
    
    def instantiate(
        self,
        template_name: str,
        bindings: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Instantiate a template with specific slot bindings.
        
        Args:
            template_name: Name of template to instantiate
            bindings: Dict mapping slot names to HDC vectors
        
        Returns:
            Resulting HDC vector, or None if template not found
        
        Example:
            >>> result = lib.instantiate("rotate", {
            ...     "INPUT": grid_vec,
            ...     "ANGLE": angle_90_vec
            ... })
        """
        template = self.templates.get(template_name)
        if template is None:
            return None
        
        # Check required slots
        for slot in template.slots:
            if slot.required and slot.name not in bindings:
                if slot.default_seed is not None:
                    bindings[slot.name] = self.hdc.from_seed(slot.default_seed)
                else:
                    return None  # Missing required slot
        
        # Get prototype
        prototype = self.prototypes[template_name]
        
        # Apply operations
        result = None
        saved_results = {"_prototype": prototype}
        
        for op in template.operations:
            op_type = op["op"]
            args = op["args"]
            
            if op_type == "bind":
                left = self._resolve_arg(args[0], bindings, saved_results, result)
                right = self._resolve_arg(args[1], bindings, saved_results, result)
                if left is None or right is None:
                    continue
                result = self.hdc.bind(left, right)
            
            elif op_type == "unbind":
                left = self._resolve_arg(args[0], bindings, saved_results, result)
                right = self._resolve_arg(args[1], bindings, saved_results, result)
                if left is None or right is None:
                    continue
                result = self.hdc.unbind(left, right)
            
            elif op_type == "bundle":
                vecs = []
                for arg in args:
                    v = self._resolve_arg(arg, bindings, saved_results, result)
                    if v is not None:
                        vecs.append(v)
                if vecs:
                    result = bind_all(vecs, self.hdc)
            
            elif op_type == "permute":
                vec = self._resolve_arg(args[0], bindings, saved_results, result)
                shift = args[1] if isinstance(args[1], int) else 1
                if vec is not None:
                    result = self.hdc.permute(vec, shift)
            
            elif op_type == "invert":
                vec = self._resolve_arg(args[0], bindings, saved_results, result)
                if vec is not None:
                    result = self.hdc.invert(vec)
            
            # Save intermediate result if specified
            if "save_as" in op and result is not None:
                saved_results[op["save_as"]] = result.copy()
        
        return result
    
    def _resolve_arg(
        self,
        arg: str,
        bindings: Dict[str, np.ndarray],
        saved: Dict[str, np.ndarray],
        current_result: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Resolve an argument to a vector."""
        if arg == "_result":
            return current_result
        if arg in bindings:
            return bindings[arg]
        if arg in saved:
            return saved[arg]
        # Try to find as template prototype
        if arg in self.prototypes:
            return self.prototypes[arg]
        return None
    
    # =========================================================================
    # Dynamic Template Creation
    # =========================================================================
    
    def create_composite_template(
        self,
        name: str,
        base_templates: List[str],
        composition_type: str = "sequence",
        description: str = ""
    ) -> Template:
        """
        Create a new template by composing existing ones.
        
        Args:
            name: Name for new template
            base_templates: List of template names to compose
            composition_type: "sequence" (T2(T1(x))) or "parallel" (bundle)
            description: Description of the composite
        
        Returns:
            The new template
        """
        # Collect all slots from base templates
        all_slots = [TemplateSlot("INPUT", role="input")]
        seen_slots = {"INPUT"}
        
        for i, t_name in enumerate(base_templates):
            base = self.templates.get(t_name)
            if base:
                for slot in base.slots:
                    if slot.name != "INPUT" and slot.name not in seen_slots:
                        # Rename to avoid collision
                        new_name = f"{slot.name}_{i}" if slot.name in seen_slots else slot.name
                        all_slots.append(TemplateSlot(
                            name=new_name,
                            role=slot.role,
                            required=slot.required
                        ))
                        seen_slots.add(new_name)
        
        # Build operations
        operations = []
        if composition_type == "sequence":
            for i, t_name in enumerate(base_templates):
                if i == 0:
                    operations.append({"op": "bind", "args": ["INPUT", f"T{i}"]})
                else:
                    operations.append({"op": "bind", "args": ["_result", f"T{i}"]})
        elif composition_type == "parallel":
            for i, t_name in enumerate(base_templates):
                operations.append({
                    "op": "bind",
                    "args": ["INPUT", f"T{i}"],
                    "save_as": f"r{i}"
                })
            operations.append({
                "op": "bundle",
                "args": [f"r{i}" for i in range(len(base_templates))]
            })
        
        operations.append({"op": "bind", "args": ["_result", "_prototype"]})
        
        template = Template(
            name=name,
            slots=all_slots,
            complexity=TemplateComplexity.COMPOUND,
            description=description or f"Composite: {' + '.join(base_templates)}",
            operations=operations,
            sub_templates=base_templates
        )
        
        self.add_template(template)
        return template
    
    def create_conditional_template(
        self,
        name: str,
        condition_template: str,
        then_template: str,
        else_template: Optional[str] = None,
        description: str = ""
    ) -> Template:
        """
        Create a conditional template: IF condition THEN action ELSE other.
        
        Args:
            name: Name for new template
            condition_template: Template for condition
            then_template: Template to apply if condition matches
            else_template: Optional template for else case
            description: Description
        
        Returns:
            The new conditional template
        """
        slots = [
            TemplateSlot("INPUT", role="input"),
            TemplateSlot("CONDITION", role="modifier"),
            TemplateSlot("THEN_PARAMS", role="modifier", required=False),
        ]
        if else_template:
            slots.append(TemplateSlot("ELSE_PARAMS", role="modifier", required=False))
        
        operations = [
            {"op": "bind", "args": ["INPUT", "CONDITION"]},
            {"op": "bind", "args": ["_result", "_prototype"]}
        ]
        
        template = Template(
            name=name,
            slots=slots,
            complexity=TemplateComplexity.CONDITIONAL,
            description=description or f"IF {condition_template} THEN {then_template}",
            operations=operations,
            sub_templates=[condition_template, then_template] + ([else_template] if else_template else [])
        )
        
        self.add_template(template)
        return template
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict:
        """Get library statistics."""
        complexity_counts = {}
        for t in self.templates.values():
            c = t.complexity.name
            complexity_counts[c] = complexity_counts.get(c, 0) + 1
        
        return {
            'total_templates': len(self.templates),
            'by_complexity': complexity_counts,
            'total_prototype_bytes': len(self.prototypes) * self.hdc.byte_size
        }


class DynamicTemplateComposer:
    """
    Dynamically compose templates based on task analysis.
    
    Given a task vector, this class:
    1. Finds matching templates
    2. Tries to decompose into sub-problems
    3. Composes solutions compositionally
    
    This is the "compositional dynamic creation" system.
    """
    
    def __init__(self, library: TemplateLibrary, hdc: SparseBinaryHDC):
        self.library = library
        self.hdc = hdc
    
    def decompose_task(
        self,
        task_vec: np.ndarray,
        max_depth: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Decompose a task into sub-tasks via template matching.
        
        Tries increasingly complex templates until a match is found.
        
        Args:
            task_vec: The task vector
            max_depth: Maximum decomposition depth
        
        Returns:
            List of (template_name, similarity) for the decomposition
        """
        decomposition = []
        
        # Try each complexity level
        for complexity in TemplateComplexity:
            matches = self.library.match_template(
                task_vec,
                top_k=3,
                min_similarity=0.6,
                max_complexity=complexity
            )
            
            if matches:
                decomposition.extend(matches)
                break  # Found at this level
        
        return decomposition
    
    def compose_solution(
        self,
        template_names: List[str],
        input_vec: np.ndarray,
        modifiers: Dict[str, np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Compose a solution from multiple templates.
        
        Args:
            template_names: Templates to apply in sequence
            input_vec: Input vector
            modifiers: Additional slot bindings
        
        Returns:
            Resulting vector
        """
        result = input_vec
        modifiers = modifiers or {}
        
        for t_name in template_names:
            bindings = {"INPUT": result}
            bindings.update(modifiers)
            
            new_result = self.library.instantiate(t_name, bindings)
            if new_result is not None:
                result = new_result
        
        return result
    
    def try_all_atomic(
        self,
        input_vec: np.ndarray,
        target_vec: np.ndarray,
        modifier_options: Dict[str, List[np.ndarray]]
    ) -> Optional[Tuple[str, Dict[str, np.ndarray], float]]:
        """
        Try all atomic templates to find one that transforms input to target.
        
        Args:
            input_vec: Input vector
            target_vec: Target output vector
            modifier_options: Dict of slot name → list of possible values
        
        Returns:
            (template_name, best_bindings, similarity) or None
        """
        best_match = None
        best_sim = 0.0
        best_bindings = None
        
        for t_name in self.library.list_templates(TemplateComplexity.ATOMIC):
            template = self.library.get_template(t_name)
            if template is None:
                continue
            
            # Get modifier slots
            mod_slots = [s.name for s in template.slots if s.role == "modifier"]
            
            # Try all combinations of modifiers
            for bindings in self._generate_bindings(mod_slots, modifier_options):
                bindings["INPUT"] = input_vec
                
                result = self.library.instantiate(t_name, bindings)
                if result is not None:
                    sim = self.hdc.similarity(result, target_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = t_name
                        best_bindings = bindings.copy()
        
        if best_match and best_sim > 0.6:
            return (best_match, best_bindings, best_sim)
        return None
    
    def _generate_bindings(
        self,
        slots: List[str],
        options: Dict[str, List[np.ndarray]]
    ):
        """Generate all combinations of slot bindings."""
        if not slots:
            yield {}
            return
        
        first = slots[0]
        rest = slots[1:]
        
        first_options = options.get(first, [])
        if not first_options:
            for rest_bindings in self._generate_bindings(rest, options):
                yield rest_bindings
        else:
            for opt in first_options:
                for rest_bindings in self._generate_bindings(rest, options):
                    yield {first: opt, **rest_bindings}


if __name__ == '__main__':
    from .hdc_sparse_core import create_sparse_hdc
    
    print("=== Template Library Demo ===\n")
    
    hdc, vocab = create_sparse_hdc(dim=32768)
    library = TemplateLibrary(hdc)
    
    print(f"Library Stats: {library.get_stats()}\n")
    
    # Create some test vectors
    input_vec = hdc.random_vector(seed=100)
    angle_90 = vocab.get('rotate_90')
    
    # Instantiate template
    print("=== Template Instantiation ===")
    result = library.instantiate("rotate", {
        "INPUT": input_vec,
        "ANGLE": angle_90
    })
    print(f"rotate(input, 90) created: {result is not None}")
    print(f"Similarity to input: {hdc.similarity(result, input_vec):.4f}")
    
    # Match template
    print("\n=== Template Matching ===")
    matches = library.match_template(result, top_k=5)
    print("Top matches for result vector:")
    for name, sim in matches:
        print(f"  {name}: {sim:.4f}")
    
    # Composite template
    print("\n=== Composite Template ===")
    composite = library.create_composite_template(
        "rotate_then_flip",
        ["rotate", "flip"],
        composition_type="sequence"
    )
    print(f"Created: {composite.name}")
    print(f"Slots: {composite.slot_names()}")
    
    # Dynamic composition
    print("\n=== Dynamic Composition ===")
    composer = DynamicTemplateComposer(library, hdc)
    decomp = composer.decompose_task(result)
    print(f"Decomposition: {decomp}")
    
    print("\n✅ Demo complete!")
