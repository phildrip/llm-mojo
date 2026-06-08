#!/usr/bin/env python3
import os
import sys
import struct

def test_hsaco_segments():
    hsaco_path = "/home/phil/git/llm-mojo/scratch/q8_matmul.hsaco"
    if not os.path.exists(hsaco_path):
        print(f"Error: HSACO file not found at {hsaco_path}")
        sys.exit(1)
        
    with open(hsaco_path, "rb") as f:
        data = f.read()
        
    # Verify ELF Magic
    if data[:4] != b"\x7fELF":
        raise ValueError("Invalid ELF magic number")
        
    # Read ELF 64-bit Header elements:
    # e_phoff (Program Header table offset): offset 32, size 8 (uint64)
    # e_phnum (Number of Program Headers): offset 56, size 2 (uint16)
    # e_phentsize (Size of each Program Header): offset 54, size 2 (uint16)
    e_phoff = struct.unpack_from("<Q", data, 32)[0]
    e_phentsize = struct.unpack_from("<H", data, 54)[0]
    e_phnum = struct.unpack_from("<H", data, 56)[0]
    
    print(f"ELF Program Headers: Offset={e_phoff}, Size={e_phentsize}, Count={e_phnum}")
    
    segments = []
    # Parse program headers
    for i in range(e_phnum):
        ph_offset = e_phoff + i * e_phentsize
        # Program Header format (64-bit ELF):
        # p_type: uint32 (offset 0)
        # p_flags: uint32 (offset 4)
        # p_offset: uint64 (offset 8)
        # p_vaddr: uint64 (offset 16)
        # p_paddr: uint64 (offset 24)
        # p_filesz: uint64 (offset 32)
        # p_memsz: uint64 (offset 40)
        # p_align: uint64 (offset 48)
        p_type = struct.unpack_from("<I", data, ph_offset)[0]
        p_flags = struct.unpack_from("<I", data, ph_offset + 4)[0]
        p_offset = struct.unpack_from("<Q", data, ph_offset + 8)[0]
        p_vaddr = struct.unpack_from("<Q", data, ph_offset + 16)[0]
        p_filesz = struct.unpack_from("<Q", data, ph_offset + 32)[0]
        p_memsz = struct.unpack_from("<Q", data, ph_offset + 40)[0]
        
        if p_type == 1: # PT_LOAD
            segments.append({
                "index": i,
                "flags": p_flags,
                "offset": p_offset,
                "vaddr": p_vaddr,
                "filesz": p_filesz,
                "memsz": p_memsz
            })
            print(f"  Segment {len(segments)} (PT_LOAD): offset={hex(p_offset)}, vaddr={hex(p_vaddr)}, filesz={hex(p_filesz)}, memsz={hex(p_memsz)}, flags={p_flags}")

    # We expect 3 LOAD segments
    if len(segments) < 3:
        raise ValueError(f"Expected at least 3 PT_LOAD segments, found {len(segments)}")
        
    seg1 = segments[0] # RO (typically)
    seg2 = segments[1] # RX
    seg3 = segments[2] # RW
    
    # -------------------------------------------------------------------------
    # Rule 1: Segment 1 (RO Segment) must cover up to 0xa80 (covering KD descriptor)
    # -------------------------------------------------------------------------
    print("\n--- Verifying Segment 1 (RO Segment) ---")
    print(f"Found Segment 1: offset={hex(seg1['offset'])}, filesz={hex(seg1['filesz'])}")
    if seg1['offset'] != 0:
        raise ValueError(f"Segment 1 offset must be 0x0, found {hex(seg1['offset'])}")
    if seg1['vaddr'] != 0:
        raise ValueError(f"Segment 1 vaddr must be 0x0, found {hex(seg1['vaddr'])}")
    if seg1['filesz'] < 0xa80:
        raise ValueError(f"Segment 1 filesz is too small! Must be at least 0xa80 to cover Kernel Descriptor. Found {hex(seg1['filesz'])}")
    print("-> Segment 1 is VALID!")
    
    # -------------------------------------------------------------------------
    # Rule 2: Segment 2 (RX Segment - Instructions)
    # -------------------------------------------------------------------------
    print("\n--- Verifying Segment 2 (RX Segment - Instructions) ---")
    print(f"Found Segment 2: offset={hex(seg2['offset'])}, vaddr={hex(seg2['vaddr'])}, filesz={hex(seg2['filesz'])}")
    if seg2['offset'] != 0xb00:
        raise ValueError(f"Segment 2 offset must be 0xb00, found {hex(seg2['offset'])}")
    if seg2['vaddr'] != 0x1b00:
        raise ValueError(f"Segment 2 vaddr must be 0x1b00, found {hex(seg2['vaddr'])}")
    if seg2['filesz'] < 0x414:
        raise ValueError(f"Segment 2 filesz must be at least 0x414, found {hex(seg2['filesz'])}")
    print("-> Segment 2 is VALID!")
    
    # -------------------------------------------------------------------------
    # Rule 3: Segment 3 (RW Segment - Data)
    # -------------------------------------------------------------------------
    print("\n--- Verifying Segment 3 (RW Segment) ---")
    print(f"Found Segment 3: offset={hex(seg3['offset'])}, vaddr={hex(seg3['vaddr'])}, filesz={hex(seg3['filesz'])}")
    if seg3['offset'] != 0xf18:
        raise ValueError(f"Segment 3 offset must be 0xf18, found {hex(seg3['offset'])}")
    if seg3['vaddr'] != 0x2f18:
        raise ValueError(f"Segment 3 vaddr must be 0x2f18, found {hex(seg3['vaddr'])}")
    if seg3['filesz'] < 0x70:
        raise ValueError(f"Segment 3 filesz must be at least 0x70, found {hex(seg3['filesz'])}")
    print("-> Segment 3 is VALID!")
    
    # -------------------------------------------------------------------------
    # Rule 4: Verify Kernel Descriptor (KD) at offset 0xa40
    # -------------------------------------------------------------------------
    print("\n--- Verifying Kernel Descriptor at 0xa40 ---")
    kd_offset = 0xa40
    # Kernel descriptor fields:
    # kernel_code_entry_byte_offset (uint64 at offset 16)
    # compute_pgm_rsrc1 (uint32 at offset 48)
    # compute_pgm_rsrc2 (uint32 at offset 52)
    entry_offset = struct.unpack_from("<q", data, kd_offset + 16)[0]
    rsrc1 = struct.unpack_from("<I", data, kd_offset + 48)[0]
    rsrc2 = struct.unpack_from("<I", data, kd_offset + 52)[0]
    
    print(f"Kernel Code Entry Offset: {entry_offset}")
    print(f"COMPUTE_PGM_RSRC1: {hex(rsrc1)}")
    print(f"COMPUTE_PGM_RSRC2: {hex(rsrc2)}")
    
    if entry_offset <= 0:
        raise ValueError(f"Invalid Kernel Code Entry Offset: {entry_offset}")
    if rsrc1 == 0:
        raise ValueError("COMPUTE_PGM_RSRC1 cannot be zero")
    if rsrc2 == 0:
        raise ValueError("COMPUTE_PGM_RSRC2 cannot be zero")
    print("-> Kernel Descriptor is VALID!")
    
    print("\n=======================================================")
    print("   ALL HSACO ELF SEGMENT VALIDATION TESTS PASSED!      ")
    print("=======================================================")

if __name__ == "__main__":
    try:
        test_hsaco_segments()
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
