[ressource](https://www.linkedin.com/posts/arnavgrg_now-you-can-reduce-memory-pressure-during-activity-7110741699412271104-Gpkv/)


Now, you can reduce memory pressure during quantized fine-tuning with ease. Ludwig 0.8.4 supports a range of paged and 8-bit optimizers, including Adam, Paged Adam, Paged Adam 8-bit, AdamW, Paged AdamW, and Paged AdamW 8-bit. These optimizers are a game-changer for dealing with frequent out of memory errors during LLM fine-tuning. 📉

Why Paged Optimizers Matter
Traditional optimizers, such as Adam and SGD, can consume a significant amount of GPU memory during model training because they take up a large amount of space per parameter in the model. High memory consumption by optimizers can impact the maximum sequence lengths that can be effectively trained during fine-tuning, particularly when working with large transformer-based models. Paged optimizers are the answer! Paged Optimizers use NVIDIA Unified Memory 3 which does automatic page-to-page transfers between the CPU and GPU for error-free GPU processing in the scenario where the GPU occasionally runs out of memory. The feature works like regular memory paging between CPU RAM and the disk. Paged optimizers allocate paged memory for the optimizer states which are then automatically evicted to CPU RAM when the GPU runs out of memory and paged back into GPU memory when the memory is needed in the optimizer update step. This efficient memory management can significantly improve your training workflow. 🧮

8-bit Optimizers
8-bit optimizers use lower precision to store the optimizer state, reducing the state representations from 32-bit to 8-bit, further reducing the memory they take up on your GPU. This means you can reduce the memory used by optimizer states by nearly 75%! 💪

With Ludwig, you can use either paged optimizers or 8-bit optimizers individually or even combine their power together! 🤝

Enabling these versions is as simple as setting your optimizer type to adamw_8bit, paged_adamw, or paged_adamw_8bit in your Ludwig config:

trainer:
optimizer:
   type: paged_adamw

Upgrade to Ludwig 0.8.4 today and supercharge your LLM fine-tuning! 🚀

- Get started with Ludwig: pip install ludwig
- Ludwig GitHub: https://lnkd.in/ee4pyDBR
- Full commit: https://lnkd.in/egeq2pzm
- Read more about paged optimizers (taken from the original QLoRA paper): https://lnkd.in/eukZ4fjJ

![paged](asset/paged.jpg)