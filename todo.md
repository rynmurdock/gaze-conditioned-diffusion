

[x] use for guiding
  [x] train scanpath-conditioned model so the image will generate with relevant portions
      at your latest fixations.
        [x] morph generations and/or only update on long saccades
  [] train point-conditioned video model (longcat? ltx?) so that gaze naturally moves media
  [x] could cache generated images/video by coordinate
  [x] detect saccades?
[x] use for evading gaze
[] use for optimizing contents
  - would cite Reben
  [] using simple shape morphers
  [] using zoom-in and propogate
  [] using preference-prior-esque genrec system


[x] salience maps dataloading (https://arxiv.org/abs/1505.03581)
[x] klein training
    [x] ensure we can load klein & lora/finetune/etc. *period*.
    [x] remove text experts
    - could enforce a maximum number of points on the scanpath?
      [x] it's just RoPE values, so not a compute optimization.
    [x] patch pipe and setup qual eval
    [x] ckpt loading (see other notimplementederror)
    [x] condition on scanpath (not just fixation points)
    - if you don't keep-distill train, would need to set a timestep schedule


[x] klein keep-distill training (https://arxiv.org/abs/2605.05204)
  [x] cache teacher outputs
    - vae latents saved as tensor with mapping from images
  - may be overfitting worse by only single trajectories per image
    [x] could use LoRA & switch on/off for student / teacher
  [x] qlora, train with adam, etc.
  [] support batch_size > 1
  [x] option to add back text encoder
    [x] can fall back to diffusers klein+pipe & patch the RoPE
  [x] use existing ckpt for guiding (boxes below) to proof out further changes.
  - put onto vast.ai for full finetune?
[x] inference with saccade updating our image
[x] train at low-ish-res but at AR that matches monitor
  [x] regenerate teacher i/o pairs
[x] K timesteps instead of full T
[x] condition on points as edit image instead of RoPE
[x] could look cool to predict the large/small shrinking & appearing fixation points as well
  - already does this a bit implicitly!
[] light cfg on with/without gaze rope?

[x] Wandb-esque html
  [x] I don't need even tensorboard just the last val, train plots & images
[x] copy config to saved ckpt

[x] smoothing over scanpath
[] scanpath statistics (e.g. average length)
[x] compare the case where the art piece hides from you versus the one that follows
  [x] hides from you (with our changing latent at least) is super frustrating, as one might expect
[x] remove grey padding on data
  [] may still be some on vertical; worth bundling with scanpath stats, looking more at data
[] further Klein speedup with vllm/llamaccp/etc.?
[] the scanpath masked for diffedit should keep latents *that you've looked at*
    not at the location in the future.


[] update image after sevral ticks, not every frame
  [] update such that all fixation points are either consistent or as far as possible
    [] can you differentiate?



