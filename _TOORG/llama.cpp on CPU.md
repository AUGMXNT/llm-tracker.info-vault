https://aws.amazon.com/ec2/instance-types/c7a/
c7a.metal-48xl
192 vCPU
384 GiB RAM
https://instances.vantage.sh/aws/ec2/c7a.metal-48xl

Similar to EPYC 9654
- https://www.amd.com/en/products/cpu/amd-epyc-9654
- https://www.amazon.com/EPYC-9654-Processor-2-4GHz-100-000000789/dp/B0CNFS63SQ
- $4400

See also: https://dev.to/dkechag/google-cloud-c3d-review-record-breaking-performance-with-epyc-genoa-g13

2 x socket (`dmidecode`) confirms
`grep DIMM | grep Channel` shows 12 Channels per socket
= 24 channels
```
Theoretical Max Bandwidth = 2400 MHz * 12 * 2 * 8 bytes = 460,800 MB/s = 460.8 GB/s

Total Theoretical Max Bandwidth = 460.8 GB/s * 2 = 921.6 GB/s
```


```
$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04 LTS
Release:        24.04
Codename:       noble
```

```
$ inxi -a
CPU: 2x 96-core AMD EPYC 9R14 (-MCP SMP-) speed/min/max: 2617/1500/3702 MHz
Kernel: 6.8.0-1008-aws x86_64 Up: 10m Mem: 5.98/376.04 GiB (1.6%) Storage: 100 GiB (28.9% used)
Procs: 1889 Shell: Bash 5.2.21 inxi: 3.3.34

$ inxi -F
System:
  Host: ip-172-31-51-195 Kernel: 6.8.0-1008-aws arch: x86_64 bits: 64
  Console: pty pts/4 Distro: Ubuntu 24.04 LTS (Noble Numbat)
Machine:
  Type: Server System: Amazon EC2 product: c7a.metal-48xl v: FRUV0.10 serial: <superuser required>
  Mobo: Amazon EC2 model: N/A serial: <superuser required> BIOS: Amazon EC2 v: 1.0
    date: 10/16/2017
CPU:
  Info: 2x 96-core model: AMD EPYC 9R14 bits: 64 type: MCP SMP cache: L2: 2x 96 MiB (192 MiB)
  Speed (MHz): avg: 2645 min/max: 1500/3702 cores: 1: 2600 2: 2600 3: 2600 4: 2600 5: 2600
    6: 2600 7: 2600 8: 2600 9: 2600 10: 2600 11: 2600 12: 2600 13: 2600 14: 2600 15: 2600 16: 2600
    17: 2600 18: 2600 19: 2600 20: 2600 21: 2600 22: 2600 23: 2600 24: 2600 25: 2600 26: 2600
    27: 2600 28: 2600 29: 2600 30: 2600 31: 2600 32: 2600 33: 2600 34: 2600 35: 2600 36: 2600
    37: 2600 38: 2600 39: 2600 40: 2600 41: 2600 42: 2600 43: 2600 44: 2600 45: 2600 46: 2600
    47: 2600 48: 2600 49: 2600 50: 2600 51: 2600 52: 2600 53: 2600 54: 2600 55: 2600 56: 2600
    57: 2600 58: 2600 59: 2600 60: 2600 61: 2600 62: 3700 63: 2600 64: 2600 65: 2600 66: 2600
    67: 2600 68: 2600 69: 2600 70: 2600 71: 2600 72: 2600 73: 2600 74: 2600 75: 2600 76: 2600
    77: 2600 78: 2600 79: 2600 80: 2600 81: 2600 82: 2600 83: 2600 84: 2600 85: 2600 86: 2600
    87: 2600 88: 2600 89: 2600 90: 2600 91: 2600 92: 2600 93: 2600 94: 2600 95: 2600 96: 2600
    97: 2600 98: 2600 99: 2600 100: 2600 101: 2600 102: 2600 103: 2600 104: 2600 105: 2600
    106: 2600 107: 2600 108: 2600 109: 2600 110: 2600 111: 2600 112: 2600 113: 2600 114: 2600
    115: 2600 116: 2600 117: 3700 118: 2600 119: 2600 120: 2600 121: 2600 122: 2600 123: 3698
    124: 2600 125: 3699 126: 3699 127: 2600 128: 2600 129: 2600 130: 2600 131: 2600 132: 2600
    133: 3700 134: 2600 135: 2600 136: 2600 137: 2600 138: 2600 139: 2600 140: 2600 141: 2600
    142: 2600 143: 2600 144: 2600 145: 2600 146: 2600 147: 2600 148: 2600 149: 2600 150: 2600
    151: 2600 152: 2600 153: 2600 154: 2600 155: 2600 156: 2600 157: 3700 158: 2600 159: 2600
    160: 2600 161: 2600 162: 2600 163: 2600 164: 2600 165: 2600 166: 2600 167: 2600 168: 2600
    169: 2600 170: 2600 171: 2600 172: 2600 173: 2600 174: 2600 175: 2600 176: 2600 177: 2600
    178: 2600 179: 2600 180: 2600 181: 2600 182: 2600 183: 2600 184: 2600 185: 2600 186: 3699
    187: 2600 188: 2600 189: 2600 190: 2600 191: 2600 192: 2600
Graphics:
  Message: No PCI device data found.
  Display: server: No display server data found. Headless machine? tty: 144x52
  API: EGL v: 1.5 drivers: swrast platforms: surfaceless,device
  API: OpenGL v: 4.5 vendor: mesa v: 24.0.5-1ubuntu1 note: console (EGL sourced)
    renderer: llvmpipe (LLVM 17.0.6 256 bits)
Audio:
  Message: No device data found.
Network:
  Device-1: Amazon.com Elastic Network Adapter driver: ena
  IF: enp160s0 state: up speed: N/A duplex: N/A mac: 0e:8c:bf:06:fa:39
Drives:
  Local Storage: total: 100 GiB used: 31.89 GiB (31.9%)
  ID-1: /dev/nvme0n1 model: Amazon Elastic Block Store size: 100 GiB
Partition:
  ID-1: / size: 95.82 GiB used: 31.81 GiB (33.2%) fs: ext4 dev: /dev/nvme0n1p1
  ID-2: /boot size: 880.4 MiB used: 75.1 MiB (8.5%) fs: ext4 dev: /dev/nvme0n1p16
  ID-3: /boot/efi size: 104.3 MiB used: 6.1 MiB (5.8%) fs: vfat dev: /dev/nvme0n1p15
Swap:
  Alert: No swap data was found.
Sensors:
  Src: lm-sensors+/sys Message: No sensor data found using /sys/class/hwmon or lm-sensors.
Info:
  Memory: total: 384 GiB available: 376.04 GiB used: 5.94 GiB (1.6%)
  Processes: 1875 Uptime: 11m Init: systemd target: graphical (5) Shell: Bash inxi: 3.3.34

```



```
$ mbw 4096
Long uses 8 bytes. Allocating 2*536870912 elements = 8589934592 bytes of memory.
Using 262144 bytes as blocks for memcpy block copy test.
Getting down to business... Doing 10 runs per test.
0       Method: MEMCPY  Elapsed: 0.28007        MiB: 4096.00000 Copy: 14625.124 MiB/s
1       Method: MEMCPY  Elapsed: 0.28080        MiB: 4096.00000 Copy: 14586.895 MiB/s
2       Method: MEMCPY  Elapsed: 0.28156        MiB: 4096.00000 Copy: 14547.521 MiB/s
3       Method: MEMCPY  Elapsed: 0.27115        MiB: 4096.00000 Copy: 15106.141 MiB/s
4       Method: MEMCPY  Elapsed: 0.28118        MiB: 4096.00000 Copy: 14567.181 MiB/s
5       Method: MEMCPY  Elapsed: 0.27163        MiB: 4096.00000 Copy: 15079.558 MiB/s
6       Method: MEMCPY  Elapsed: 0.28023        MiB: 4096.00000 Copy: 14616.565 MiB/s
7       Method: MEMCPY  Elapsed: 0.28098        MiB: 4096.00000 Copy: 14577.706 MiB/s
8       Method: MEMCPY  Elapsed: 0.27197        MiB: 4096.00000 Copy: 15060.651 MiB/s
9       Method: MEMCPY  Elapsed: 0.28209        MiB: 4096.00000 Copy: 14520.240 MiB/s
AVG     Method: MEMCPY  Elapsed: 0.27816        MiB: 4096.00000 Copy: 14725.110 MiB/s
0       Method: DUMB    Elapsed: 0.24858        MiB: 4096.00000 Copy: 16477.526 MiB/s
1       Method: DUMB    Elapsed: 0.23958        MiB: 4096.00000 Copy: 17096.871 MiB/s
2       Method: DUMB    Elapsed: 0.23964        MiB: 4096.00000 Copy: 17092.091 MiB/s
3       Method: DUMB    Elapsed: 0.23965        MiB: 4096.00000 Copy: 17091.735 MiB/s
4       Method: DUMB    Elapsed: 0.23958        MiB: 4096.00000 Copy: 17096.300 MiB/s
5       Method: DUMB    Elapsed: 0.23963        MiB: 4096.00000 Copy: 17093.090 MiB/s
6       Method: DUMB    Elapsed: 0.24866        MiB: 4096.00000 Copy: 16472.490 MiB/s
7       Method: DUMB    Elapsed: 0.24847        MiB: 4096.00000 Copy: 16485.087 MiB/s
8       Method: DUMB    Elapsed: 0.25036        MiB: 4096.00000 Copy: 16360.572 MiB/s
9       Method: DUMB    Elapsed: 0.23968        MiB: 4096.00000 Copy: 17089.595 MiB/s
AVG     Method: DUMB    Elapsed: 0.24338        MiB: 4096.00000 Copy: 16829.504 MiB/s
0       Method: MCBLOCK Elapsed: 0.24047        MiB: 4096.00000 Copy: 17033.097 MiB/s
1       Method: MCBLOCK Elapsed: 0.23170        MiB: 4096.00000 Copy: 17678.108 MiB/s
2       Method: MCBLOCK Elapsed: 0.23198        MiB: 4096.00000 Copy: 17656.466 MiB/s
3       Method: MCBLOCK Elapsed: 0.23220        MiB: 4096.00000 Copy: 17639.662 MiB/s
4       Method: MCBLOCK Elapsed: 0.23189        MiB: 4096.00000 Copy: 17663.319 MiB/s
5       Method: MCBLOCK Elapsed: 0.23249        MiB: 4096.00000 Copy: 17618.114 MiB/s
6       Method: MCBLOCK Elapsed: 0.24058        MiB: 4096.00000 Copy: 17025.168 MiB/s
7       Method: MCBLOCK Elapsed: 0.23484        MiB: 4096.00000 Copy: 17441.365 MiB/s
8       Method: MCBLOCK Elapsed: 0.24221        MiB: 4096.00000 Copy: 16910.945 MiB/s
9       Method: MCBLOCK Elapsed: 0.23224        MiB: 4096.00000 Copy: 17637.231 MiB/s
AVG     Method: MCBLOCK Elapsed: 0.23506        MiB: 4096.00000 Copy: 17425.227 MiB/s
```

```
wget https://panthema.net/2013/pmbw/pmbw-0.6.2.tar.bz2
tar xvf pmbw-0.6.2.tar.bz2
cd pmbw-0.6.2
./configure

$ ./pmbw -M 68719476736 -p 96 -P 96
Setting memory limit to 68719476736.                                                 Running benchmarks with at least 96 threads.
Running benchmarks with up to 96 threads.
CPUID: mmx sse avx
Detected 385060 MiB physical RAM and 192 CPUs.
Allocating 32768 MiB for testing.

...

2720280675349.9350586

```

```
# https://github.com/rsusik/rambenchmark
sudo apt install python3-pip
pip install rambenchmark --break-system-packages
$ ~/.local/bin/rambenchmark                                                                                                       
======================================================================                                                                                                 
BENCHMARKING RAM WITH MULTI THREADS                                                                                                                                    
(...please wait...)                                                                                                                                                    
                                                                                   
192 concurrent threads are supported.                                              
                                                                                   
----------------------------------------------------------------------                                                                                                 
MEMSET TEST                                                                                                                                                            
                                                                                                                                                                       
RESULT of filling 1GiB buffer with zeros.                                          
>>> 0.0012 (s) / 925165.9 (MB/s) <<<                                               
                                                                                                                                                                       
                   Details                                                                                                                                             
  #Threads        Time (s)      Speed (MB/s)                                                                                                                           
         1      0.0321 (s)    33460.1 (MB/s)                                       
         2      0.0168 (s)    63957.8 (MB/s)                                       
         3      0.0122 (s)    87647.8 (MB/s)                                                                                                                           
         4      0.0151 (s)    71218.3 (MB/s)                                                                                                                           
         5      0.0119 (s)    89911.9 (MB/s)                                                                                                                           
         6      0.0100 (s)   107088.6 (MB/s)                                       
         7      0.0086 (s)   124667.3 (MB/s)                                                                                                                           
         8      0.0076 (s)   141231.3 (MB/s)                                                                                                                           
         9      0.0068 (s)   157202.5 (MB/s)                                                                                                                           
        10      0.0062 (s)   172765.5 (MB/s)
           11      0.0057 (s)   189002.9 (MB/s)                                       
        12      0.0053 (s)   201940.7 (MB/s)                                       
        13      0.0050 (s)   213970.5 (MB/s)                                       
        14      0.0048 (s)   222553.3 (MB/s)                                       
        15      0.0045 (s)   238250.8 (MB/s)                                       
        16      0.0043 (s)   250022.8 (MB/s)                                       
        17      0.0040 (s)   266743.1 (MB/s)                                       
        18      0.0039 (s)   273965.8 (MB/s)                                       
        19      0.0036 (s)   297225.2 (MB/s)                                       
        20      0.0034 (s)   316145.6 (MB/s)                                       
        21      0.0032 (s)   339490.8 (MB/s)                                       
        22      0.0029 (s)   365649.4 (MB/s)                                       
        23      0.0027 (s)   391826.8 (MB/s)                                       
        24      0.0026 (s)   409303.7 (MB/s)                                       
        25      0.0030 (s)   357352.8 (MB/s)                                       
        26      0.0039 (s)   275024.7 (MB/s)                                       
        27      0.0036 (s)   296586.9 (MB/s)                                       
        28      0.0035 (s)   303183.9 (MB/s)                                       
        29      0.0034 (s)   316592.4 (MB/s)                                       
        30      0.0033 (s)   326964.1 (MB/s)                                       
        31      0.0032 (s)   335873.4 (MB/s)                                       
        32      0.0031 (s)   350541.8 (MB/s)                                       
        33      0.0030 (s)   353888.9 (MB/s)                                       
        34      0.0028 (s)   388808.3 (MB/s)                                       
        35      0.0028 (s)   383191.7 (MB/s)                                       
        36      0.0029 (s)   371990.1 (MB/s)                                       
        37      0.0030 (s)   362927.3 (MB/s)                                       
        38      0.0028 (s)   387670.4 (MB/s)                                       
        39      0.0026 (s)   406135.6 (MB/s)                                       
        40      0.0026 (s)   407243.9 (MB/s)                                       
        41      0.0027 (s)   403228.1 (MB/s)                                       
        42      0.0027 (s)   401798.2 (MB/s)                                       
        43      0.0026 (s)   414209.7 (MB/s)                                       
        44      0.0026 (s)   419092.3 (MB/s)                                       
        45      0.0029 (s)   371820.0 (MB/s)                                       
        46      0.0025 (s)   437207.0 (MB/s)                                       
        47      0.0025 (s)   430042.0 (MB/s)                                       
        48      0.0027 (s)   391821.1 (MB/s)                                       
        49      0.0025 (s)   428129.0 (MB/s)                                       
        50      0.0024 (s)   450885.0 (MB/s)                                       
        51      0.0027 (s)   400311.1 (MB/s)                                       
        52      0.0027 (s)   404729.3 (MB/s)                                       
        53      0.0027 (s)   393493.6 (MB/s)                                       
        54      0.0026 (s)   405317.9 (MB/s)                                   
                55      0.0027 (s)   393812.8 (MB/s)                                                                                                        15:29:09 [297/3121]
        56      0.0026 (s)   415436.6 (MB/s)                                       
        57      0.0027 (s)   399019.0 (MB/s)                                       
        58      0.0025 (s)   435703.3 (MB/s)                                       
        59      0.0024 (s)   453074.5 (MB/s)                                       
        60      0.0028 (s)   383236.9 (MB/s)                                       
        61      0.0022 (s)   477418.5 (MB/s)                                       
        62      0.0021 (s)   519231.9 (MB/s)                                       
        63      0.0021 (s)   504966.4 (MB/s)                                       
        64      0.0021 (s)   511126.7 (MB/s)                                       
        65      0.0020 (s)   535823.8 (MB/s)                                       
        66      0.0020 (s)   536032.6 (MB/s)                                       
        67      0.0024 (s)   450323.0 (MB/s)                                       
        68      0.0023 (s)   475569.1 (MB/s)                                       
        69      0.0024 (s)   447926.2 (MB/s)                                       
        70      0.0023 (s)   469351.1 (MB/s)                                       
        71      0.0022 (s)   480251.9 (MB/s)                                       
        72      0.0021 (s)   504569.8 (MB/s)                                       
        73      0.0024 (s)   455511.4 (MB/s)                                       
        74      0.0024 (s)   448115.1 (MB/s)                                       
        75      0.0021 (s)   505606.4 (MB/s)                                       
        76      0.0024 (s)   448268.6 (MB/s)                                       
        77      0.0027 (s)   400788.1 (MB/s)                                       
        78      0.0023 (s)   460046.3 (MB/s)                                       
        79      0.0023 (s)   460524.1 (MB/s)                                       
        80      0.0022 (s)   481402.5 (MB/s)                                       
        81      0.0022 (s)   480973.1 (MB/s)                                       
        82      0.0021 (s)   505642.2 (MB/s)                                       
        83      0.0021 (s)   508575.6 (MB/s)                                       
        84      0.0021 (s)   516363.6 (MB/s)                                       
        85      0.0022 (s)   489620.3 (MB/s)                                       
        86      0.0021 (s)   499252.3 (MB/s)                                       
        87      0.0021 (s)   514711.2 (MB/s)                                       
        88      0.0020 (s)   535840.1 (MB/s)                                       
        89      0.0020 (s)   532360.1 (MB/s)                                       
        90      0.0020 (s)   524454.8 (MB/s)                                       
        91      0.0028 (s)   378135.8 (MB/s)                                       
        92      0.0022 (s)   484688.3 (MB/s)                                       
        93      0.0019 (s)   572617.0 (MB/s)                                       
        94      0.0017 (s)   624101.0 (MB/s)                                       
        95      0.0017 (s)   622544.0 (MB/s)                                       
        96      0.0017 (s)   636703.4 (MB/s)                                       
        97      0.0019 (s)   563291.9 (MB/s)                                       
        98      0.0019 (s)   567501.5 (MB/s)                                       
        99      0.0021 (s)   515628.9 (MB/s)                                       
        100      0.0021 (s)   517803.9 (MB/s)                                       
       101      0.0022 (s)   494448.6 (MB/s)                                       
       102      0.0023 (s)   469416.8 (MB/s)                                       
       103      0.0022 (s)   483772.4 (MB/s)                                       
       104      0.0018 (s)   593610.4 (MB/s)                                       
       105      0.0016 (s)   652413.3 (MB/s)                                       
       106      0.0015 (s)   697444.8 (MB/s)                                       
       107      0.0017 (s)   639641.2 (MB/s)                                       
       108      0.0016 (s)   668214.4 (MB/s)                                       
       109      0.0017 (s)   614258.0 (MB/s)                                       
       110      0.0019 (s)   563040.6 (MB/s)                                       
       111      0.0020 (s)   528128.1 (MB/s)                                       
       112      0.0020 (s)   535575.1 (MB/s)                                       
       113      0.0022 (s)   488223.3 (MB/s)                                       
       114      0.0021 (s)   506644.9 (MB/s)                                       
       115      0.0018 (s)   608323.9 (MB/s)                                       
       116      0.0019 (s)   573756.0 (MB/s)                                       
       117      0.0016 (s)   672565.8 (MB/s)                                       
       118      0.0016 (s)   655585.9 (MB/s)                                       
       119      0.0017 (s)   649581.4 (MB/s)                                       
       120      0.0016 (s)   657704.0 (MB/s)                                       
       121      0.0016 (s)   657264.9 (MB/s)                                       
       122      0.0018 (s)   585090.9 (MB/s)                                       
       123      0.0020 (s)   545963.2 (MB/s)                                       
       124      0.0021 (s)   522122.6 (MB/s)                                       
       125      0.0021 (s)   511238.7 (MB/s)                                       
       126      0.0021 (s)   518980.8 (MB/s)                                       
       127      0.0022 (s)   483879.6 (MB/s)                                       
       128      0.0022 (s)   490874.7 (MB/s)                                       
       129      0.0018 (s)   584485.1 (MB/s)                                       
       130      0.0017 (s)   629534.9 (MB/s)                                       
       131      0.0016 (s)   655425.7 (MB/s)                                       
       132      0.0017 (s)   642817.9 (MB/s)                                       
       133      0.0016 (s)   654690.2 (MB/s)                                       
       134      0.0016 (s)   680888.1 (MB/s)                                       
       135      0.0018 (s)   580664.5 (MB/s)                                       
       136      0.0018 (s)   606097.2 (MB/s)                                       
       137      0.0018 (s)   601132.6 (MB/s)                                       
       138      0.0018 (s)   583779.4 (MB/s)                                                              139      0.0018 (s)   590337.9 (MB/s)                                                                                                        15:29:13 [213/3121]
       140      0.0019 (s)   572567.8 (MB/s)                                       
       141      0.0019 (s)   564584.1 (MB/s)                                       
       142      0.0020 (s)   534327.2 (MB/s)                                       
       143      0.0021 (s)   521582.0 (MB/s)                                       
       144      0.0020 (s)   531234.7 (MB/s)                                       
       145      0.0022 (s)   498456.8 (MB/s)                                       
       146      0.0021 (s)   501865.3 (MB/s)                                       
       147      0.0019 (s)   550645.3 (MB/s)                                       
       148      0.0018 (s)   610758.1 (MB/s)                                       
       149      0.0017 (s)   640817.8 (MB/s)                                       
       150      0.0017 (s)   647277.3 (MB/s)                                       
       151      0.0016 (s)   656528.9 (MB/s)                                       
       152      0.0016 (s)   658139.7 (MB/s)                                       
       153      0.0016 (s)   662860.0 (MB/s)                                       
       154      0.0015 (s)   704718.9 (MB/s)                                       
       155      0.0015 (s)   704011.4 (MB/s)                                       
       156      0.0015 (s)   717484.4 (MB/s)                                       
       157      0.0016 (s)   680145.8 (MB/s)                                       
       158      0.0015 (s)   702357.5 (MB/s)                                       
       159      0.0015 (s)   705729.3 (MB/s)                                       
       160      0.0015 (s)   698484.5 (MB/s)                                       
       161      0.0016 (s)   677538.9 (MB/s)                                       
       162      0.0016 (s)   684580.6 (MB/s)                                       
       163      0.0016 (s)   688163.3 (MB/s)                                       
       164      0.0016 (s)   675810.6 (MB/s)                                       
       165      0.0017 (s)   618022.4 (MB/s)                                       
       166      0.0017 (s)   642402.3 (MB/s)                                       
       167      0.0017 (s)   626619.5 (MB/s)                                       
       168      0.0018 (s)   611666.8 (MB/s)                                       
       169      0.0018 (s)   604955.9 (MB/s)                                       
       170      0.0019 (s)   575553.0 (MB/s)                                       
       171      0.0017 (s)   615982.4 (MB/s)                                       
       172      0.0015 (s)   704316.4 (MB/s)                                       
       173      0.0014 (s)   779574.4 (MB/s)                                       
       174      0.0013 (s)   829043.9 (MB/s)                                       
       175      0.0013 (s)   839060.0 (MB/s)                                       
       176      0.0013 (s)   835252.2 (MB/s)                                       
       177      0.0013 (s)   831910.6 (MB/s)                                       
       178      0.0013 (s)   843923.7 (MB/s)                                       
       179      0.0012 (s)   889141.5 (MB/s)                                       
       180      0.0012 (s)   877299.1 (MB/s)                                       
       181      0.0012 (s)   899261.0 (MB/s)                                       
       182      0.0012 (s)   898613.3 (MB/s)                                       
       183      0.0012 (s)   921677.2 (MB/s)                                                     184      0.0012 (s)   902263.0 (MB/s)                                       
       185      0.0012 (s)   886892.7 (MB/s)                                       
       186      0.0012 (s)   922097.0 (MB/s)                                       
       187      0.0012 (s)   925165.9 (MB/s)                                       
       188      0.0012 (s)   924050.5 (MB/s)                                       
       189      0.0012 (s)   906715.6 (MB/s)                                       
       190      0.0012 (s)   888427.4 (MB/s)                                       
       191      0.0013 (s)   839270.0 (MB/s)                                       
       192      0.0012 (s)   863093.2 (MB/s)         

		  ----------------------------------------------------------------------                                                                                                 
MEMCHR TEST                                                                        

RESULT of scanning 1GiB buffer.                                                    
>>> 0.0009 (s) / 1190171.2 (MB/s) <<<                                              


                   Details                                                         
  #Threads        Time (s)      Speed (MB/s)                                       
         1      0.0333 (s)    32194.9 (MB/s)                                       
         2      0.0171 (s)    62675.1 (MB/s)                                       
         3      0.0115 (s)    93078.2 (MB/s)                                       
         4      0.0094 (s)   113785.6 (MB/s)                                       
         5      0.0080 (s)   134926.4 (MB/s)                                       
         6      0.0068 (s)   158773.3 (MB/s)                                       
         7      0.0058 (s)   184677.5 (MB/s)                                       
         8      0.0048 (s)   225058.1 (MB/s)                                       
         9      0.0047 (s)   228628.3 (MB/s)                                       
        10      0.0043 (s)   247119.6 (MB/s)                                       
        11      0.0042 (s)   257431.0 (MB/s)                                       
        12      0.0040 (s)   271157.0 (MB/s)         
    
        96      0.0019 (s)   562916.8 (MB/s)                                                                                                                           
        97      0.0019 (s)   570137.2 (MB/s)                                                                                                                           
        98      0.0019 (s)   565110.7 (MB/s)                                                                                                                           
        99      0.0019 (s)   571035.1 (MB/s)                                                                                                                           
       100      0.0018 (s)   580250.0 (MB/s)                          


       160      0.0010 (s)  1111339.0 (MB/s)                                       
       161      0.0009 (s)  1133875.5 (MB/s)                                       
       162      0.0009 (s)  1143262.6 (MB/s)                                       
       163      0.0009 (s)  1148562.8 (MB/s)                                       
       164      0.0009 (s)  1132045.2 (MB/s)                                       
       165      0.0009 (s)  1138954.6 (MB/s)                                       
       166      0.0011 (s)  1019068.8 (MB/s)                                       
       167      0.0011 (s)   977799.5 (MB/s)                                       
       168      0.0010 (s)  1085558.8 (MB/s)                                       
       169      0.0011 (s)   946131.4 (MB/s)                                       
       170      0.0012 (s)   900952.3 (MB/s)                                       
       171      0.0014 (s)   781072.6 (MB/s)                                       
       172      0.0010 (s)  1118509.9 (MB/s)                                       
       173      0.0009 (s)  1190171.2 (MB/s)                                       
       174      0.0009 (s)  1140602.4 (MB/s)                                       
       175      0.0009 (s)  1163393.3 (MB/s)                                       
       176      0.0009 (s)  1176211.4 (MB/s)                                       
       177      0.0010 (s)  1063374.3 (MB/s)                                       
       178      0.0009 (s)  1144642.0 (MB/s)                                       
       179      0.0010 (s)  1108995.8 (MB/s)                                       
       180      0.0011 (s)  1021115.1 (MB/s)                                       
       181      0.0011 (s)  1018682.8 (MB/s)                                       
       182      0.0011 (s)   948153.8 (MB/s)                                       
       183      0.0012 (s)   903166.0 (MB/s)                                       
       184      0.0012 (s)   894455.9 (MB/s)                                       
       185      0.0011 (s)   948758.2 (MB/s)                                       
       186      0.0011 (s)   934659.2 (MB/s)                                       
       187      0.0012 (s)   911227.9 (MB/s)                                       
       188      0.0012 (s)   916317.1 (MB/s)                                       
       189      0.0013 (s)   832039.7 (MB/s)                                       
       190      0.0014 (s)   784006.0 (MB/s)                                       
       191      0.0013 (s)   827732.8 (MB/s)                                       
       192      0.0016 (s)   675036.8 (MB/s)                                       


					 

```


llama2-7b q4_0
```
$ time ./llama-bench -p 3968 -m llama-2-7b.Q4_0.gguf 
| model                          |       size |     params | backend    |    threads | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ---------: | ---------- | ---------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CPU        |         96 | pp 3968    |    200.22 ± 1.39 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | CPU        |         96 | tg 128     |     17.81 ± 0.01 |

build: 928e0b70 (2749)

real    2m38.418s
user    208m2.475s
sys     39m6.294s
```

llama3-70b q4_k_m
```
$ time llama.cpp$ ./llama-bench -p 3968 -m Meta-Llama-3-70B-Q4_K_M.gguf 
| model                          |       size |     params | backend    |    threads | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ---------: | ---------- | ---------------: |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | CPU        |         96 | pp 3968    |     21.00 ± 0.29 |
| llama 70B Q4_K - Medium        |  39.59 GiB |    70.55 B | CPU        |         96 | tg 128     |      2.85 ± 0.00 |

build: 928e0b70 (2749)

real    22m46.328s
user    1712m25.681s
sys     459m23.131s
```