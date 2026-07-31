import torch
import logging
import urllib.request, os


def rng_proper_codename():
    # Source - https://stackoverflow.com/a/49524775
    # Posted by amoodie
    CACHE_FILE = "words_cache.txt"
    word_url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'} 

    if not os.path.exists(CACHE_FILE):
        req = urllib.request.Request(word_url, headers=headers) 
        long_txt = urllib.request.urlopen(req).read().decode()
        open(CACHE_FILE, 'w').write(long_txt)
        logging.info(f'Saving word list starting with: {long_txt[:100]=}')

    words = open(CACHE_FILE).read().splitlines()
    upper_words = [word for word in words if word[0].isupper()]
    name_words  = [word for word in upper_words if not word.isupper()]
    rand_name   = ' '.join([name_words[torch.randint(0, len(name_words), (1,))] for i in range(2)])
    return rand_name

def setup_log_dir(config):
    # random code name or possibly a name set by config set by hparam sweep script
    codename = rng_proper_codename().replace(' ', '_') if not config.exp_name else config.exp_name
    dir_name = f'./logs/{codename}/'
    # we always want a parent dir
    os.makedirs('./logs/', exist_ok=True)
    # but will die before replacing an existing experiment log dir
    os.makedirs(dir_name, exist_ok=False)
    config.to_json(os.path.join(dir_name, 'config.json'))
    return dir_name

if __name__ == "__main__":
    logging.info(rng_proper_codename())



