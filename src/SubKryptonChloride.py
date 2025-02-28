import matplotlib.pyplot as plt

wavelengths = [
    190.0, 190.5, 191.0, 191.5, 192.0, 192.5, 193.0, 193.5, 194.0, 194.5,
    195.0, 195.5, 196.0, 196.5, 197.0, 197.5, 198.0, 198.5, 199.0, 199.5,
    200.0, 200.5, 201.0, 201.5, 202.0, 202.5, 203.0, 203.5, 204.0, 204.5,
    205.0, 205.5, 206.0, 206.5, 207.0, 207.5, 208.0, 208.5, 209.0, 209.5,
    210.0, 210.5, 211.0, 211.5, 212.0, 212.5, 213.0, 213.5, 214.0, 214.5,
    215.0, 215.5, 216.0, 216.5, 217.0, 217.5, 218.0, 218.5, 219.0, 219.5,
    220.0, 220.5, 221.0, 221.5, 222.0, 222.5, 223.0, 223.5, 224.0, 224.5,
    225.0, 225.5, 226.0, 226.5, 227.0, 227.5, 228.0, 228.5, 229.0, 229.5,
    230.0, 230.5, 231.0, 231.5, 232.0, 232.5, 233.0, 233.5, 234.0, 234.5,
    235.0, 235.5, 236.0, 236.5, 237.0, 237.5, 238.0, 238.5, 239.0, 239.5,
    240.0, 240.5, 241.0, 241.5, 242.0, 242.5, 243.0, 243.5, 244.0, 244.5,
    245.0, 245.5, 246.0, 246.5, 247.0, 247.5, 248.0, 248.5, 249.0, 249.5,
    250.0, 250.5, 251.0, 251.5, 252.0, 252.5, 253.0, 253.5, 254.0, 254.5,
    255.0, 255.5, 256.0, 256.5, 257.0, 257.5, 258.0, 258.5, 259.0, 259.5,
    260.0, 260.5, 261.0, 261.5, 262.0, 262.5, 263.0, 263.5, 264.0, 264.5,
    265.0, 265.5, 266.0, 266.5, 267.0, 267.5, 268.0, 268.5, 269.0, 269.5,
    270.0, 270.5, 271.0, 271.5, 272.0, 272.5, 273.0, 273.5, 274.0, 274.5,
    275.0
]

intensities = [
    0.0, 100.0, 100.0, 88.5, 84.0, 98.3, 13.3, 100.0, 85.5, 74.1,
    100.0, 81.3, 64.7, 53.7, 44.0, 36.4, 30.2, 25.1, 21.3, 18.1,
    15.1, 12.7, 11.2, 9.4, 8.1, 7.2, 6.4, 5.7, 5.2, 4.7,
    4.3, 3.9, 3.5, 3.3, 3.1, 2.8, 2.6, 2.4, 2.3, 2.1,
    2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.5, 1.4, 1.4, 1.3,
    1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2,
    1.4, 1.6, 1.8, 2.0, 2.3, 2.7, 3.1, 3.7, 4.7, 5.3,
    6.8, 8.0, 9.4, 10.5, 11.7, 13.0, 14.3, 15.3, 16.1, 15.9,
    15.5, 14.2, 12.4, 10.8, 9.5, 8.4, 7.6, 6.8, 6.2, 5.6,
    5.1, 4.7, 4.3, 3.9, 3.5, 3.2, 2.9, 2.6, 2.4, 2.2,
    2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.5, 1.4, 1.4, 1.4,
    1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1,
    1.1, 1.1, 1.1, 1.2, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8,
    2.0, 2.2, 2.4, 2.7, 3.0, 3.5, 4.0, 5.1, 6.8, 8.9,
    11.1, 13.4, 15.0, 15.8, 15.3, 13.8, 12.3, 10.8, 9.6, 8.6,
    7.7, 7.0, 6.4, 5.8, 5.3, 4.8, 4.5, 4.1, 3.8, 3.4,
    3.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0
]

plt.figure(figsize=(8, 5))
plt.semilogy(wavelengths, intensities, 'b.-', label='KrCl Data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Intensity (log scale)')
plt.title('Krypton Chloride Emission Spectrum (Extracted)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Optionally adjust the y-axis limits if you want to see more detail in the low-intensity region.
# plt.ylim(bottom=1e-1)

plt.show()
