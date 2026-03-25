//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS OS Primitives
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Essential structures and traits for building an Operating System kernel
//! on top of AEGIS. These primitives provide the "Data Points" and "Fall Backs"
//! needed for hardware interaction, context switching, and error handling.
//!
//! Target Architecture: x86_64
//!
//! ═══════════════════════════════════════════════════════════════════════════════

/// CPU Register Context
///
/// This structure represents the state of the CPU registers. It is used for:
/// - Context Switching (threads/processes)
/// - Exception Handling (saving state before handling interrupt)
/// - System Calls (passing arguments)
///
/// Layout matches the "System V AMD64 ABI" calling convention and common
/// interrupt stack frame layouts.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CpuContext {
    // General Purpose Registers
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rbp: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rbx: u64,
    pub rax: u64,

    // Instruction Pointer (saved by interrupt/call)
    // Included here for complete context restoration
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

impl CpuContext {
    /// Create a new, blank CPU context
    pub const fn empty() -> Self {
        Self {
            r15: 0, r14: 0, r13: 0, r12: 0, r11: 0, r10: 0, r9: 0, r8: 0,
            rbp: 0, rdi: 0, rsi: 0, rdx: 0, rcx: 0, rbx: 0, rax: 0,
            rip: 0, cs: 0, rflags: 0, rsp: 0, ss: 0,
        }
    }
}

/// Interrupt Exception Frame
///
/// This is the data structure pushed onto the stack by the CPU when an
/// interrupt or exception occurs in 64-bit mode.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ExceptionFrame {
    /// Instruction Pointer (Return Address)
    pub rip: u64,
    /// Code Segment Selector
    pub cs: u64,
    /// CPU Flags (Condition codes)
    pub rflags: u64,
    /// Stack Pointer
    pub rsp: u64,
    /// Stack Segment Selector
    pub ss: u64,
}

/// Page Table Entry (PTE)
///
/// Represents a single entry in x86_64 4-level paging structure.
/// Provides a type-safe wrapper around raw bits.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct PageTableEntry {
    pub entry: u64,
}

impl PageTableEntry {
    // Standard x86_64 Paging Bits
    const PRESENT: u64 = 1 << 0;
    const WRITABLE: u64 = 1 << 1;
    const USER_ACCESSIBLE: u64 = 1 << 2;
    const WRITE_THROUGH: u64 = 1 << 3;
    const NO_CACHE: u64 = 1 << 4;
    const ACCESSED: u64 = 1 << 5;
    const DIRTY: u64 = 1 << 6;
    const HUGE_PAGE: u64 = 1 << 7;
    const GLOBAL: u64 = 1 << 8;
    const NO_EXECUTE: u64 = 1 << 63;

    /// Create a raw entry
    pub const fn new(entry: u64) -> Self {
        Self { entry }
    }

    /// Is the page present in memory?
    pub fn is_present(&self) -> bool {
        (self.entry & Self::PRESENT) != 0
    }

    /// Is the page writable?
    pub fn is_writable(&self) -> bool {
        (self.entry & Self::WRITABLE) != 0
    }

    /// Get the physical address (masking out flags)
    pub fn addr(&self) -> u64 {
        self.entry & 0x000F_FFFF_FFFF_F000
    }
    
    /// Set the physical address
    pub fn set_addr(&mut self, addr: u64) {
         let flags = self.entry & !0x000F_FFFF_FFFF_F000;
         self.entry = (addr & 0x000F_FFFF_FFFF_F000) | flags;
    }
}

/// Operating System Hooks
///
/// Interface for the kernel to hook into AEGIS core events.
/// Implement this trait to provide OS-specific handling.
pub trait OsHooks {
    /// Called when a kernel panic occurs.
    /// Should dump registers and halt implementation.
    fn on_panic(&self, info: &core::panic::PanicInfo);

    /// Called to log a message (e.g., to serial port or VGA).
    fn log(&self, message: &str);

    /// Called before entering a sleep state (for power management).
    fn on_sleep(&self);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bio-Kernel Hardware Adaptability Layer
// ═══════════════════════════════════════════════════════════════════════════════

/// Physical address wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct PhysAddr(pub u64);

/// Virtual address wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct VirtAddr(pub u64);

/// Memory region type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryType {
    Usable,
    Reserved,
    AcpiReclaimable,
    AcpiNvs,
    BadMemory,
    KernelCode,
    KernelStack,
    PageTable,
    FrameBuffer,
    BootloaderInfo,
}

/// A contiguous region of physical memory
#[derive(Debug, Clone, Copy)]
pub struct MemoryRegion {
    pub start: PhysAddr,
    pub length: u64,
    pub region_type: MemoryType,
}

/// Hardware Topology (The "Body" Layout)
///
/// This structure represents the discovered physical reality of the machine.
/// The kernel uses this to adapt its "biological" functions (scheduling, memory).
#[derive(Debug, Clone)]
pub struct HardwareTopology {
    /// Number of physical CPU cores (Neural Clusters)
    pub cpu_cores: usize,
    
    /// Number of NUMA nodes (Memory Locality Domains)
    pub numa_nodes: usize,
    
    /// Total usable system memory in bytes
    pub total_memory: u64,
    
    /// Physical address of the root configuration (RSDP for ACPI, DTB for ARM)
    pub config_root: PhysAddr,
    
    /// List of memory regions (The "Territory")
    pub memory_map: [MemoryRegion; 32], // Fixed size for no_std bootstrap
    
    /// Number of valid regions in the map
    pub memory_map_len: usize,
}

impl HardwareTopology {
    /// Create a new, empty topology
    pub fn new() -> Self {
        Self {
            cpu_cores: 1, // Assume 1 core (Safe Mode) until probed
            numa_nodes: 1,
            total_memory: 0,
            config_root: PhysAddr(0),
            memory_map: [MemoryRegion {
                start: PhysAddr(0),
                length: 0,
                region_type: MemoryType::Reserved,
            }; 32],
            memory_map_len: 0,
        }
    }

    /// Add a memory region found during Bio-Scan
    pub fn add_region(&mut self, region: MemoryRegion) {
        if self.memory_map_len < 32 {
            self.memory_map[self.memory_map_len] = region;
            self.memory_map_len += 1;
            
            if region.region_type == MemoryType::Usable {
                self.total_memory += region.length;
            }
        }
    }

    /// Adaptivity: Determine optimal mode based on hardware
    pub fn suggest_mode(&self) -> KernelMode {
        if self.cpu_cores > 4 && self.total_memory > 1024 * 1024 * 1024 * 4 {
            KernelMode::DeepManifold // High parallelism, deep history
        } else if self.cpu_cores > 1 {
            KernelMode::Standard // Standard multi-core
        } else {
            KernelMode::SafeSerial // Single core, minimal memory
        }
    }
}

/// Operational mode of the kernel, adapted to hardware
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelMode {
    /// Minimal footprint, single threaded
    SafeSerial,
    /// Standard operation
    Standard,
    /// High-performance geometric optimizations enabled
    DeepManifold,
}

/// Interface that the Bootloader/BIOS layer must provide
pub trait BiosInterface {
    /// Get the raw memory map from firmware
    fn memory_map(&self) -> &[MemoryRegion];
    
    /// Get the framebuffer info (if graphics enabled)
    fn framebuffer(&self) -> Option<FrameBufferInfo>;
    
    /// Get the ACPI/DTB root pointer
    fn config_root(&self) -> PhysAddr;
}

#[derive(Debug, Clone, Copy)]
pub struct FrameBufferInfo {
    pub address: PhysAddr,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub bytes_per_pixel: u8,
}

