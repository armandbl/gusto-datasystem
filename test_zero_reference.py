#!/usr/bin/env python
import sys
sys.path.insert(0, r'D:\gusto-datasystem\src\GUSTO_Pipeline')

from L10_pointing import getMixerOffsets
import argparse

# Create a mock args class
class MockArgs:
    def __init__(self, zero_reference=False):
        self.zero_reference = zero_reference

# Test without zero-reference
print("Testing WITHOUT zero-reference:")
args_normal = MockArgs(zero_reference=False)
offsets_normal = getMixerOffsets(1, [1,2,3,4,5,6,7,8], args=args_normal)
print("Offsets for band 1, mixers 1-8:")
print("AZ:", offsets_normal['az'])
print("EL:", offsets_normal['el'])

# Test with zero-reference
print("\nTesting WITH zero-reference:")
args_zero = MockArgs(zero_reference=True)
offsets_zero = getMixerOffsets(1, [1,2,3,4,5,6,7,8], args=args_zero)
print("Offsets for band 1, mixers 1-8 (zero-referenced):")
print("AZ:", offsets_zero['az'])
print("EL:", offsets_zero['el'])

# Check that the first mixer (reference) is now at (0,0)
print("\nReference mixer (first in list) offsets:")
print("AZ:", offsets_zero['az'][0])
print("EL:", offsets_zero['el'][0])

# Also test band 2
print("\n\nBand 2 test:")
offsets_normal_b2 = getMixerOffsets(2, [1,2,3,4,5,6,7,8], args=MockArgs(zero_reference=False))
offsets_zero_b2 = getMixerOffsets(2, [1,2,3,4,5,6,7,8], args=MockArgs(zero_reference=True))
print("Band 2 normal AZ:", offsets_normal_b2['az'])
print("Band 2 zero-referenced AZ:", offsets_zero_b2['az'])
print("Band 2 zero-referenced EL:", offsets_zero_b2['el'])
print("Band 2 reference mixer (first) AZ:", offsets_zero_b2['az'][0])
print("Band 2 reference mixer (first) EL:", offsets_zero_b2['el'][0])