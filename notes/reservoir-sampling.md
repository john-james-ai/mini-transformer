Reservoir sampling is a family of algorithms designed to randomly select a fixed-size subset (a "reservoir") of elements from a stream of data, where the total number of elements in the stream (N) is often unknown or too large to store in memory. The key characteristic is that each element in the original stream has an equal probability of being included in the final reservoir. [1]  
Core Principles: 

• Initialization: The reservoir is initially filled with the first k elements from the stream, where k is the desired size of the sample. 
• Processing Subsequent Elements: For each subsequent element (the i-th element, where i &gt; k): [2]  
	• A random number is generated. 
	• If this random number falls within a specific probability range (e.g., k/i), the i-th element is selected. 
	• If selected, one of the elements currently in the reservoir is randomly chosen and replaced with the i-th element. 

• Uniform Probability: The algorithm ensures that at any point in the stream, every element processed so far has an equal probability of being in the reservoir. This is achieved by carefully calculating the probabilities of inclusion and replacement. 

Why it's Useful: 

• Memory Efficiency: It offers a highly memory-efficient way to sample large datasets or data streams, as it only requires O(k) space (where k is the reservoir size), regardless of the stream's total length. 
• Handling Unknown Stream Sizes: It can be applied even when the total number of elements (N) in the stream is unknown in advance, which is common in real-time data processing. 
• Applications: It finds use in various domains, including: 
	• Data Preprocessing: Creating representative subsets for analysis or model training from large datasets. 
	• Real-time Analytics: Monitoring and analyzing data streams in applications like network traffic analysis or financial transaction monitoring. 
	• Random Selection: Selecting a random subset from an unknown-length sequence. 

AI responses may include mistakes.

Here’s the reservoir sampler in plain English—what each piece does and why it’s correct.

# Goal

From a one-pass stream of unknown/huge length $N$, pick **exactly $k$** items so that **every item has equal chance $k/N$** of being in the final sample. Use only $O(k)$ memory.

# The moving parts

* **`iterable`**: your data stream (e.g., HuggingFace `IterableDataset`).
* **`k`**: desired sample size (your 120).
* **`R`**: the “reservoir,” a list that will hold up to `k` items.
* **`i`**: 1-based index of the current item in the stream.
* **`j`**: a random integer in $[0, i-1]$ (or $[1,i]$ if you like 1-based). This is the key lottery.

# The procedure

1. **Fill phase (first $k$ items):**
   For the first $k$ items ($i \le k$), simply append them to `R`. After this, `R` is full.

2. **Replacement phase (every item after $k$):**
   For item $i = k+1, k+2, \dots$:

   * Draw **one** random integer $j$ uniformly from $[0, i-1]$ (i.e., each of the first $i$ positions is equally likely).
   * **If $j < k$**, replace `R[j]` with the new item; otherwise discard the new item.

That’s it.

# Why it’s uniform (proof sketch)

Consider any fixed position $t$ in the stream:

* If $t \le k$:
  Item $t$ is definitely in the reservoir after the fill phase. From step $s = k+1$ to $N$, it **survives** each later step with probability $(1 - 1/s)$, because at step $s$ we replace a reservoir entry with probability $k/s$ and each current member has chance $1/k$ to be the one replaced → replacement chance $ (k/s)\cdot(1/k)=1/s$.
  Survival probability across all later steps:

  $$
  \prod_{s=k+1}^{N}\left(1 - \frac{1}{s}\right) = \frac{k}{N}.
  $$

  So final inclusion probability is $1 \cdot \frac{k}{N} = \frac{k}{N}$.

* If $t > k$:
  At its own step $t$, it’s included with probability $k/t$ (since $j<k$ with probability $k/t$). Then it must **survive** replacements from $s=t+1$ to $N$, each with factor $(1 - 1/s)$.
  Final probability:

  $$
  \frac{k}{t} \cdot \prod_{s=t+1}^{N}\left(1 - \frac{1}{s}\right) = \frac{k}{t} \cdot \frac{t}{N} = \frac{k}{N}.
  $$

In both cases, every item ends with the same inclusion probability $k/N$. That’s exact uniform sampling.

# Complexity & behavior

* **One pass**, **$O(k)$ memory**.
* **Time:** $O(N)$ with a constant-time decision per item.
* **If the stream ends early ($N<k$)**, you simply end up with $N$ items; that’s the best possible.

# Practical notes (for your use)

* Use a **fixed seed** if you want reproducibility.
* Do any **filtering** (e.g., length limits) **before** the sampler; then the sample is uniform over the filtered set.
* Don’t mix this with approximate buffer shuffles if you want strict uniformity; just stream in native order and let the reservoir do the randomness.
* After you collect the $k$ items, you can **shuffle** the reservoir once and slice into 80/10/10 splits deterministically.

# Variants you might care about later

* **Weighted reservoir sampling** (Efraimidis–Spirakis): items with weights $w_i$ get inclusion probability proportional to weight.
* **Streaming stratified sampling**: maintain one reservoir per stratum (e.g., by length bins or domain) if you need balanced splits.

If you want, I can sanity-check your pseudocode or the proof once you write it up.


[1] https://alhardy.github.io/app-metrics-docs/getting-started/sampling/index.html[2] https://medium.com/pythoneers/dipping-into-data-streams-the-magic-of-reservoir-sampling-762f41b78781
