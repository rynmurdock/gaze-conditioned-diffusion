

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
    - if you don't keep-distill train, def would need to set a timestep schedule


[x] klein keep-distill training (https://arxiv.org/abs/2605.05204)
  [x] cache teacher outputs
    - vae latents saved as tensor with mapping from images
  - may be overfitting worse by only single trajectories per image
    [x] could use LoRA & switch on/off for student / teacher
  [x] qlora, train with adam, etc.
  [] batch_size > 1 training
    [x] remove older cached latents feature
    [x] remove older scanpath RoPE conditioning
    [x] support batch_size > 1 RoPE with test
    [x] pil images -> tensor -> vae
    [x] seq padding + attention masking with test

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
[x] light cfg on with/without gaze cond img

[x] Wandb-esque html
  [x] I don't need even tensorboard just the last val, train plots & images
[x] copy config to saved ckpt

[x] smoothing over scanpath
[x] scanpath statistics (e.g. average length)
[x] remove sketches, drawings from data (ugly, simple, bad images)
[x] compare the case where the art piece hides from you versus the one that follows
  [x] hides from you (with our changing latent at least) is super frustrating, as one might expect
[x] remove grey padding on data
  [] may still be some on vertical; worth checking+removing
[x] fix LoRA saving+loading
[] the scanpath masked for diffedit should keep latents *that you've looked at*
    not the ones at the location in the next tick.
[] further Klein speedup with vllm/llamaccp/etc.?


[] variations on famous painitngs by scanpath
  [] look through your existing data
  [] collect your own scanpaths

[] update image after several ticks, not every frame
  [] update such that all fixation points are either consistent or as far as possible
    [] can you differentiate?
[] Visual beats w/ different content
  - Mundane image w/ visual beats of art
[] we still have "change portion that isn't visually interesting" to explore re consumption <> creation
[] blank canvas/static -> 12 points & see if you can get good

hparam sweep:
  - under-trained models?
  [x] lora rank
  [x] lr

[x] compare teacher model sampling typical trajectory & with gt answer given
[x] timestep shift by resolution

[] starting writeup
[x] eval over checkpoints using lpips/clip/dinoscore
  [] just to show d-opsd is necessary, ablate with normal flow-matching loss
    - just need example image/similar to demonstrate

[] copy of .py code files of repo into log file at point of launch

[x] delete logs with only images & configs
[x] finetune on reasonable data

[] given reasonable lr & rank, interesting comparisons are:
  [] just_inf_timesteps (k or t)
    ./logs/kathemoglobin_Bonnette_Douai/ vs just_inf_version_corelates_Salzgitter_Steatornis
    - using lpips+dino; val loss won't be valid
    [] try w/ shifting timesteps but only if k is close or less-good
[] correlation between val loss & reconstruction metrics

[] dino + lpips score are thrown off by different types of images
    - i.e., a well-done product photo on white is nothing like the painting, ofc
  [] gaze -> image -> gaze/scanpath model compared to GT scanpath
  [x] train/finetune on a specific type




