"""Feature-map visualization (raw channels and PCA) for ASD models via forward hooks."""

import os
import cv2
import glob
import math
import numpy as np
import torch
import python_speech_features
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_feature_maps(
    self, 
    dataPath, 
    target_layer_names, 
    num_runs=2,
    duration=1.0,
    pca_frames=[0, 4, 9],
    save=False,
    save_folder="feature_analysis",
    save_channels=True,
    overwrite=False,
):
    """Extract and plot the feature maps of selected layers on the first face tracks of a folder.

    For each of the first ``num_runs`` tracks in ``dataPath``, the first
    ``duration`` seconds go through the model. Then, for every hooked spatial
    layer, these figures are produced: all channels of the first frame
    (optional), a PCA (3 components -> RGB) of the frames in ``pca_frames``, the
    input frames, and a combined input / PCA / overlay plot.

    Args:
        self (pyasd.Model): The :class:`~pyasd.engine.model.Model` wrapper (it is
            passed explicitly, e.g. ``visualize_feature_maps(model, ...)``).
        dataPath (str): ``pycrop`` folder with ``<track>.avi`` / ``<track>.wav`` pairs.
        target_layer_names (list[str]): Names of the modules to hook, as given
            by ``model.model.named_modules()`` (e.g. ``'visualEncoder.block1'``).
        num_runs (int): Number of tracks to process.
        duration (float): Seconds of each track fed to the model.
        pca_frames (list[int]): Frame indices used for the PCA plots.
        save (bool): Save the figures as PNGs instead of showing them.
        save_folder (str): Output folder. If it exists and ``overwrite`` is
            False, ``_1``, ``_2``, ... is appended.
        save_channels (bool): Also produce the grid of all channels of frame 0.
        overwrite (bool): Write into ``save_folder`` even if it already exists.
    """
    if hasattr(self.model, 'deploy'):
        self.model = self.model.deploy()
    else:
        self.model.eval()

    # ==========================================
    # 0. Setup Output Folder
    # ==========================================
    if save:
        if overwrite or not os.path.exists(save_folder):
            final_save_dir = save_folder
        else:
            n = 1
            while os.path.exists(f"{save_folder}_{n}"):
                n += 1
            final_save_dir = f"{save_folder}_{n}"
        
        os.makedirs(final_save_dir, exist_ok=True)
        print(f"Directory created. Saving images to: {final_save_dir}")

    # ==========================================
    # 1. Setup PyTorch Forward Hooks
    # ==========================================
    feature_maps = {}
    
    def get_features_hook(layer_name):
        """Create a forward hook that stores the output of ``layer_name`` in ``feature_maps``."""
        def hook(module, input, output):
            """Store the detached output on the CPU."""
            feature_maps[layer_name] = output.detach().cpu()
        return hook

    registered_hooks = []
    for name, module in self.model.named_modules():
        if name in target_layer_names:
            print(f"Adding hook to {name}")
            hook_handle = module.register_forward_hook(get_features_hook(name))
            registered_hooks.append(hook_handle)
            
    if not registered_hooks:
        print(f"Warning: None of the target layers {target_layer_names} were found in the model.")

    # ==========================================
    # 2. Iterate Over Files
    # ==========================================
    files = glob.glob("%s/*.avi" % dataPath)
    files.sort()
    
    for file in files[:num_runs]:
        feature_maps.clear() 
        current_fileName = os.path.splitext(file.split('/')[-1])[0] 
        print(f"Processing and visualizing: {current_fileName}...")
        
        # Load video 
        video = cv2.VideoCapture(file)
        videoFeature, fps = [], round(video.get(cv2.CAP_PROP_FPS))
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (112, 112))
                videoFeature.append(face[None])
            else:
                break
        video.release()
        videoFeature = np.stack(videoFeature, axis=0)

        # Load audio
        hzAudio = 4 * fps 
        _, audio = wavfile.read(os.path.join(dataPath, current_fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep = 13, 
            winlen = 0.025 * 25 / fps, winstep = 0.01 * 25 / fps)

        # Convert to tensors
        dtype = torch.float
        videoFeature = torch.tensor(videoFeature, device=self.device, dtype=dtype).unsqueeze(0) / 255
        audioFeature = torch.tensor(audioFeature, device=self.device, dtype=dtype).unsqueeze(0).transpose(1, 2)
        
        # Alignment logic
        length = min((audioFeature.shape[2] - audioFeature.shape[2] % 4) / hzAudio, videoFeature.shape[1])
        audioFeature = audioFeature[:, :, :int(round(length * hzAudio))]
        videoFeature = videoFeature[:, :int(round(length * fps))]
        
        target_hz_len = int(duration * hzAudio)
        target_fps_len = int(duration * fps)
        
        inputA = audioFeature[:, :, :min(target_hz_len, audioFeature.shape[2])]
        inputV = videoFeature[:, :min(target_fps_len, videoFeature.shape[1])]

        # ==========================================
        # 3. Forward Pass
        # ==========================================
        with torch.no_grad():
            _ = self.model(inputV, inputA)

        # ==========================================
        # 4. Visualization 
        # ==========================================
        for layer_name, fmap in feature_maps.items():
            print(f"  -> Layer: {layer_name} | Shape: {fmap.shape}")
            
            is_3d = fmap.dim() == 5 
            if fmap.dim() < 4:
                print(f"  -> Skipping spatial visualization for {layer_name} (Not a spatial feature map).")
                continue

            # ---------------------------------------------------------
            # PLOT A: All Feature Maps for the First Frame
            # ---------------------------------------------------------
            if is_3d:
                fmap_first_frame = fmap[0, :, 0, :, :]
            else:
                fmap_first_frame = fmap[0, :, :, :]

            num_channels = fmap_first_frame.shape[0]
            cols = math.ceil(math.sqrt(num_channels))
            rows = math.ceil(num_channels / cols)

            if not save_channels:
                pass
            elif save:
                cell_upscale_size = 112 
                cv_grid_rows = []
                idx = 0
                for r in range(rows):
                    row_images = []
                    for c in range(cols):
                        if idx < num_channels:
                            ch_arr = fmap_first_frame[idx].numpy()
                            ch_norm = cv2.normalize(ch_arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            ch_color = cv2.applyColorMap(ch_norm, cv2.COLORMAP_VIRIDIS)
                            ch_color_resized = cv2.resize(ch_color, (cell_upscale_size, cell_upscale_size), interpolation=cv2.INTER_NEAREST)
                            row_images.append(ch_color_resized)
                        else:
                            row_images.append(np.zeros((cell_upscale_size, cell_upscale_size, 3), dtype=np.uint8))
                        idx += 1
                    cv_grid_rows.append(np.hstack(row_images))
                
                final_cv_grid = np.vstack(cv_grid_rows)
                save_path = os.path.join(final_save_dir, f"{current_fileName}_{layer_name}_all_channels_frame0.png")
                cv2.imwrite(save_path, final_cv_grid)
            else:
                fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
                fig.suptitle(f"File: {current_fileName} | Layer: {layer_name} | All Channels (Frame 0)", fontsize=14)
                axes_flat = axes.flat if num_channels > 1 else [axes]
                for i, ax in enumerate(axes_flat):
                    if i < num_channels:
                        ax.imshow(fmap_first_frame[i].numpy(), cmap='viridis')
                        ax.set_title(f"Ch {i}", fontsize=8)
                    ax.axis('off')
                plt.tight_layout()
                plt.show() 

            # ---------------------------------------------------------
            # PLOT B, C, D: PCA Processing & Overlays
            # ---------------------------------------------------------
            if is_3d:
                collected_cv_inputs = []
                collected_cv_pcas = []
                collected_cv_overlays = []

                collected_plt_inputs = []
                collected_plt_pcas = []
                collected_plt_overlays = []
                
                valid_pca_frames = []

                for t_idx in pca_frames:
                    if t_idx >= fmap.shape[2]:
                        continue
                    
                    valid_pca_frames.append(t_idx)

                    fmap_t = fmap[0, :, t_idx, :, :]
                    C, H, W = fmap_t.shape
                    
                    X = fmap_t.reshape(C, -1).T.numpy() 
                    
                    pca = PCA(n_components=3)
                    X_pca = pca.fit_transform(X)
                    
                    X_pca_min = X_pca.min(axis=0)
                    X_pca_max = X_pca.max(axis=0)
                    X_pca_norm = (X_pca - X_pca_min) / (X_pca_max - X_pca_min + 1e-8)
                    
                    pca_mask = X_pca_norm.reshape(H, W, 3)

                    orig_frame = inputV[0, t_idx, 0].cpu().numpy() 
                    orig_H, orig_W = orig_frame.shape
                    
                    pca_mask_nearest = cv2.resize(pca_mask, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
                    pca_mask_nearest = np.clip(pca_mask_nearest, 0.0, 1.0)
                    
                    orig_rgb = cv2.cvtColor(np.float32(orig_frame), cv2.COLOR_GRAY2RGB)
                    plt_overlay = (orig_rgb * 0.35) + (pca_mask_nearest * 0.65)
                    
                    collected_plt_inputs.append(orig_frame)
                    collected_plt_pcas.append(pca_mask_nearest)
                    collected_plt_overlays.append(plt_overlay)

                    if save:
                        orig_bgr = cv2.cvtColor((orig_frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        pca_bgr = cv2.cvtColor((pca_mask_nearest * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        
                        alpha = 0.65
                        overlay_bgr = cv2.addWeighted(orig_bgr, 1 - alpha, pca_bgr, alpha, 0)

                        collected_cv_inputs.append(orig_bgr)
                        collected_cv_pcas.append(pca_bgr)
                        collected_cv_overlays.append(overlay_bgr)

                # =======================================================
                # GENERATE THE THREE REQUESTED IMAGES
                # =======================================================
                if len(valid_pca_frames) > 0:
                    
                    # --- IMAGE 1: ONLY PCAs STRIP ---
                    if save:
                        final_cv_pcas_only = np.hstack(collected_cv_pcas)
                        save_path_pcas = os.path.join(final_save_dir, f"{current_fileName}_{layer_name}_pcas_only.png")
                        cv2.imwrite(save_path_pcas, final_cv_pcas_only)
                    else:
                        fig1, axes1 = plt.subplots(1, len(valid_pca_frames), figsize=(4 * len(valid_pca_frames), 4))
                        fig1.suptitle(f"PCA Masks Only | Layer: {layer_name}", fontsize=14)
                        if len(valid_pca_frames) == 1: axes1 = [axes1]
                        
                        for i, ax in enumerate(axes1):
                            ax.imshow(collected_plt_pcas[i])
                            ax.set_title(f"PCA Frame {valid_pca_frames[i]}")
                            ax.axis('off')
                        plt.tight_layout()
                        plt.show()

                    # --- IMAGE 2: ONLY INPUTS STRIP (The missing piece!) ---
                    if save:
                        final_cv_inputs_only = np.hstack(collected_cv_inputs)
                        save_path_inputs = os.path.join(final_save_dir, f"{current_fileName}_{layer_name}_inputs_only.png")
                        cv2.imwrite(save_path_inputs, final_cv_inputs_only)
                    else:
                        fig_in, axes_in = plt.subplots(1, len(valid_pca_frames), figsize=(4 * len(valid_pca_frames), 4))
                        fig_in.suptitle(f"Input Frames Only | Layer: {layer_name}", fontsize=14)
                        if len(valid_pca_frames) == 1: axes_in = [axes_in]
                        
                        for i, ax in enumerate(axes_in):
                            ax.imshow(collected_plt_inputs[i], cmap='gray')
                            ax.set_title(f"Frame {valid_pca_frames[i]}")
                            ax.axis('off')
                        plt.tight_layout()
                        plt.show()

                    # --- IMAGE 3: FULL PCA PLOT (3 Rows) ---
                    if save:
                        row1 = np.hstack(collected_cv_inputs)
                        row2 = np.hstack(collected_cv_pcas)
                        row3 = np.hstack(collected_cv_overlays)
                        
                        final_cv_full_pca = np.vstack([row1, row2, row3])
                        save_path_full = os.path.join(final_save_dir, f"{current_fileName}_{layer_name}_full_pca_plot.png")
                        cv2.imwrite(save_path_full, final_cv_full_pca)
                    else:
                        fig2, axes2 = plt.subplots(3, len(valid_pca_frames), figsize=(4 * len(valid_pca_frames), 12))
                        fig2.suptitle(f"Full PCA Plot | Layer: {layer_name}", fontsize=14)
                        
                        if len(valid_pca_frames) == 1:
                            axes2 = np.expand_dims(axes2, axis=1)

                        for i in range(len(valid_pca_frames)):
                            # Row 1: Frames
                            axes2[0, i].imshow(collected_plt_inputs[i], cmap='gray')
                            axes2[0, i].set_title(f"Frame {valid_pca_frames[i]}")
                            axes2[0, i].axis('off')

                            # Row 2: PCAs
                            axes2[1, i].imshow(collected_plt_pcas[i])
                            axes2[1, i].set_title(f"PCA Frame {valid_pca_frames[i]}")
                            axes2[1, i].axis('off')

                            # Row 3: Frames + PCAs
                            axes2[2, i].imshow(collected_plt_overlays[i])
                            axes2[2, i].set_title(f"Overlay Frame {valid_pca_frames[i]}")
                            axes2[2, i].axis('off')
                            
                        plt.tight_layout()
                        plt.show()

    for hook in registered_hooks:
        hook.remove()