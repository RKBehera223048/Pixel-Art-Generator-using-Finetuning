/**
 * PIXEL_LORA — Frontend Application Logic
 * Handles image generation, gallery loading, UI interactions, and lightbox.
 */

(() => {
  "use strict";

  // ================================================================
  // DOM References
  // ================================================================
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const promptInput       = $("#prompt-input");
  const negativeInput     = $("#negative-prompt-input");
  const stepsSlider       = $("#steps-slider");
  const guidanceSlider    = $("#guidance-slider");
  const widthSlider       = $("#width-slider");
  const heightSlider      = $("#height-slider");
  const stepsVal          = $("#steps-val");
  const guidanceVal       = $("#guidance-val");
  const widthVal          = $("#width-val");
  const heightVal         = $("#height-val");
  const generateBtn       = $("#generate-btn");
  const previewArea       = $("#preview-area");
  const previewPlaceholder = $("#preview-placeholder");
  const previewImage      = $("#preview-image");
  const loadingOverlay    = $("#loading-overlay");
  const downloadBtn       = $("#download-btn");
  const genMeta           = $("#gen-meta");
  const metaTime          = $("#meta-time");
  const metaSize          = $("#meta-size");
  const metaSteps         = $("#meta-steps");
  const galleryGrid       = $("#gallery-grid");
  const marqueeTrack      = $("#marquee-track");
  const inferenceImages   = $("#inference-images");
  const lightbox          = $("#lightbox");
  const lightboxImg       = $("#lightbox-img");
  const lightboxClose     = $("#lightbox-close");
  const errorToast        = $("#error-toast");
  const mobileToggle      = $("#mobile-toggle");
  const mobileNav         = $("#mobile-nav");
  const mobileClose       = $("#mobile-close");

  // ================================================================
  // State
  // ================================================================
  let currentImageB64 = null;
  let currentFilename = null;
  let isGenerating    = false;

  // ================================================================
  // Slider live labels
  // ================================================================
  function bindSlider(slider, display, suffix = "") {
    slider.addEventListener("input", () => {
      display.textContent = slider.value + suffix;
    });
  }
  bindSlider(stepsSlider, stepsVal);
  bindSlider(guidanceSlider, guidanceVal);
  bindSlider(widthSlider, widthVal);
  bindSlider(heightSlider, heightVal);

  // ================================================================
  // Mobile Navigation
  // ================================================================
  mobileToggle.addEventListener("click", () => {
    mobileNav.classList.add("open");
  });

  mobileClose.addEventListener("click", () => {
    mobileNav.classList.remove("open");
  });

  $$(".mobile-nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      mobileNav.classList.remove("open");
    });
  });

  // ================================================================
  // Active Nav Link tracking
  // ================================================================
  const navLinks = $$(".navbar__links a");
  const sections = ["pipeline", "studio-section", "gallery-section"];

  function updateActiveNav() {
    let current = "";
    for (const id of sections) {
      const section = document.getElementById(id);
      if (section && window.scrollY >= section.offsetTop - 200) {
        current = id;
      }
    }
    navLinks.forEach((link) => {
      const href = link.getAttribute("href").replace("#", "");
      link.classList.toggle("active", href === current);
    });
  }
  window.addEventListener("scroll", updateActiveNav, { passive: true });

  // ================================================================
  // Toast Notifications
  // ================================================================
  let toastTimer = null;
  function showToast(message, duration = 5000) {
    errorToast.textContent = message;
    errorToast.classList.add("visible");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      errorToast.classList.remove("visible");
    }, duration);
  }

  // ================================================================
  // Gallery Loading
  // ================================================================
  async function loadGallery() {
    try {
      const res = await fetch("/api/gallery");
      if (!res.ok) throw new Error("Failed to load gallery");
      const images = await res.json();

      // Main gallery grid
      galleryGrid.innerHTML = "";
      if (images.length === 0) {
        galleryGrid.innerHTML = `
          <div class="gallery__empty">
            <span class="material-symbols-outlined" style="font-size:3rem;opacity:0.3;display:block;margin-bottom:1rem;">photo_library</span>
            No images yet. Generate your first pixel art above!
          </div>`;
      } else {
        images.forEach((name) => {
          const item = document.createElement("div");
          item.className = "gallery__item brutal-card";
          item.innerHTML = `
            <img src="/gallery/${name}" alt="${name}" loading="lazy" />
            <div class="gallery__item-name">${name}</div>`;
          item.addEventListener("click", () => openLightbox(`/gallery/${name}`));
          galleryGrid.appendChild(item);
        });
      }

      // Populate hero marquee (use up to 8 images, duplicated for seamless loop)
      const marqueeImages = images.slice(0, 8);
      marqueeTrack.innerHTML = "";
      if (marqueeImages.length > 0) {
        const createMarqueeSet = () => {
          marqueeImages.forEach((name) => {
            const img = document.createElement("img");
            img.src = `/gallery/${name}`;
            img.alt = name;
            img.className = "brutal-border";
            img.loading = "lazy";
            marqueeTrack.appendChild(img);
          });
        };
        createMarqueeSet(); // Original
        createMarqueeSet(); // Duplicate for seamless loop
      }

      // Populate inference showcase (use up to 4 images)
      const inferenceImgs = images.slice(0, 4);
      inferenceImages.innerHTML = "";
      if (inferenceImgs.length > 0) {
        const prompts = [
          '"pixel_art, heroic knight..."',
          '"pixel_art, cyberpunk city..."',
          '"pixel_art, wizard library..."',
          '"pixel_art, forest dragon..."',
        ];
        inferenceImgs.forEach((name, i) => {
          const wrap = document.createElement("div");
          wrap.className = "card-inference__img-wrap";
          wrap.innerHTML = `
            <img src="/gallery/${name}" alt="${name}" class="brutal-border" loading="lazy" />
            <div class="card-inference__prompt-overlay brutal-border">${prompts[i] || `"pixel_art, ${name}"`}</div>`;
          inferenceImages.appendChild(wrap);
        });
      } else {
        inferenceImages.innerHTML = `
          <div style="grid-column:1/-1;text-align:center;padding:2rem;color:var(--on-surface-variant);font-weight:600;">
            Generate some images to see them here!
          </div>`;
      }
    } catch (err) {
      console.error("Gallery load error:", err);
    }
  }

  // ================================================================
  // Lightbox
  // ================================================================
  function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    lightbox.classList.remove("open");
    document.body.style.overflow = "";
    lightboxImg.src = "";
  }

  lightboxClose.addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });

  // ================================================================
  // Image Generation
  // ================================================================
  generateBtn.addEventListener("click", async () => {
    if (isGenerating) return;

    const prompt = promptInput.value.trim();
    if (!prompt) {
      showToast("Please enter a prompt before generating.");
      promptInput.focus();
      return;
    }

    isGenerating = true;
    generateBtn.disabled = true;
    generateBtn.innerHTML = `
      <span class="material-symbols-outlined">hourglass_top</span>
      Generating...`;
    previewPlaceholder.style.display = "none";
    previewImage.style.display = "none";
    loadingOverlay.classList.add("active");
    downloadBtn.classList.remove("visible");
    genMeta.classList.remove("visible");

    const payload = {
      prompt: prompt,
      negative_prompt: negativeInput.value.trim(),
      steps: parseInt(stepsSlider.value),
      guidance_scale: parseFloat(guidanceSlider.value),
      width: parseInt(widthSlider.value),
      height: parseInt(heightSlider.value),
    };

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Generation failed.");
      }

      // Display image
      currentImageB64 = data.image;
      currentFilename = data.filename;
      previewImage.src = `data:image/png;base64,${data.image}`;
      previewImage.style.display = "block";

      // Show meta
      metaTime.textContent = `⏱ ${data.elapsed}s`;
      metaSize.textContent = `📐 ${data.params.width}×${data.params.height}`;
      metaSteps.textContent = `🔄 ${data.params.steps} steps`;
      genMeta.classList.add("visible");

      // Show download button
      downloadBtn.classList.add("visible");

      // Refresh gallery
      loadGallery();

    } catch (err) {
      console.error("Generation error:", err);
      showToast(`Error: ${err.message}`);
      previewPlaceholder.style.display = "flex";
    } finally {
      loadingOverlay.classList.remove("active");
      isGenerating = false;
      generateBtn.disabled = false;
      generateBtn.innerHTML = `
        <span class="material-symbols-outlined">auto_awesome</span>
        Generate Artwork`;
    }
  });

  // ================================================================
  // Download
  // ================================================================
  downloadBtn.addEventListener("click", () => {
    if (!currentImageB64) return;

    const link = document.createElement("a");
    link.href = `data:image/png;base64,${currentImageB64}`;
    link.download = currentFilename || "pixel_art_generated.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  // ================================================================
  // Init
  // ================================================================
  loadGallery();
})();
