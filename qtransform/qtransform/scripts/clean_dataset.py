import logging
log = logging.getLogger(__name__)

import re
import enum
import json
import glob
import os
from typing import Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import enchant
import qtransform
args = [ "run=script", "run.file=none" ]
@qtransform.with_config(args, logging.DEBUG)
def run_standalone(cfg):
	log.info(f"running {__file__}")
	file = "/home/kuhmichel/.qtransform/datasets/downloads/TinyStories_all_data"
	stats = { "errors": 0, "succcess": 0, }
	ouputs = write_outputs(file="outputs")
	next(ouputs)
	with logging_redirect_tqdm():
		for _i, e in tqdm(enumerate(gather_files(file))):
			# print(e[Keys.story])
			if kget(e,Keys.source) == Keys.gpt3:
				continue
			good_grammer, data = process_story_grammer(e)
			good_vocab, data = check_story_vocab(e) 
			yes = good_grammer and good_vocab
			if yes:
				stats['succcess'] = stats['succcess'] + 1
				ouputs.send(data[Keys.story]+"\n<|endoftext|>\n\n")
			else:
				stats['errors'] = stats['errors'] + 1
				
		#if _i > 10:
	#		break
	ouputs.close()

	print(unknown_words)
	print(stats)

class Keys(str, enum.Enum):
	story = 'story'
	instruction = 'instruction'
	source = 'source'
	instruction_prompt = 'instruction.prompt:'
	instruction_words = 'instruction.words'
	instruction_features = 'instruction.features'
	summary = 'summary'
	gpt4 = "GPT-4"
	gpt3 = "GPT-3.5"
	def __str__(self):
		return f'{str(self.value)}'
		
#pat = re.compile(r"(.+?)[\s'\",!,\.,\?,-]{1,}")
pat = re.compile(r"([\w][\w]*'?\w?)")
number = re.compile(r"^[+-]?((\d+(\.\d+)?)|(\.\d+))$")
puncts = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.',
           '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', 
           '`', '{', '|', '}', '~', '»', '«', '“', '”']
punct_pattern = re.compile("[" + re.escape("".join(puncts)) + "]{1}")
eigenNames = re.compile(r"([A-Z]{1}[\w]+)") 
unknown_words = dict()
#english = enchant.Dict("en_US")
english_us = enchant.DictWithPWL("en_US","mywords.txt")
english_gb = enchant.DictWithPWL("en_GB","mywords.txt")
blacklist  = enchant.PyPWL("blacklist.txt")
def check_story_vocab(data:str):
	story = data[Keys.story]
	story_words = pat.findall(story)
	word:str
	for word in story_words:
		if bool(number.search(word)): # skip numbers
			continue
		if blacklist.check(word):
			return False, data
		if not english_us.check(word) and not english_gb.check(word): # try and recover in case we can
			word = word.rstrip("'r")
			word = word.rstrip("'l")
			word = word.rstrip("'s")
			#word = word.rstrip("'re")
			#word = word.rstrip("'ll")
			word = word.rstrip("'v")
			#word = word.rstrip("'ve")
			if word == "": # empty after strip => always missformed word
				#print("? " + word)
				return False, data
			if blacklist.check(word): # blacklist again
				return False, data
			if english_us.check(word) or english_gb.check(word):  # words after strip
				continue
			if english_us.check(word.lower()) or english_gb.check(word.lower()): # kaptial words
				continue
			if bool(eigenNames.search(word)): # eigen namen, Fifi, Momo, Juila, Romeo, ...
				continue
			_word = word.rstrip("y") # tippy, stormy, tipsy, shacky, etc...
			if english_us.check(_word) or english_gb.check(_word):  # words after strip
				continue
			if english_us.check(_word.lower()) or english_gb.check(_word.lower()): # kaptial words
				continue
			_word = word.rstrip("ly") # tippy, stormy, tipsy, shacky, etc...
			if english_us.check(_word) or english_gb.check(_word):  # words after strip
				continue
			if english_us.check(_word.lower()) or english_gb.check(_word.lower()): # kaptial words
				continue

			if unknown_words.get(word, None) is not None:
				unknown_words[word] = unknown_words[word]+1
			else:
				unknown_words[word] = 1
			return False, data
		
	return True, data

def process_story_grammer(data:dict):
	story = kget(data, Keys.story)
	try:
		story = clean_punctuation(story)
		# checks must return True if check failed!
		#  better reporting?
		fail = check_len(story) \
			or check_ascii(story) \
			or check_odd_symbols(story) \
			or check_punct(story)
		if not fail:
			return True, kset(data, Keys.story, story)
		else:
			return False, data
	except Exception as e:
		log.warning(f"check failed {e}, story is skipped")
		return False, data

def gather_files(file):
	if isinstance(file, str):
		if os.path.isdir(file):
			log.debug(f"assume file {file} is a folder")
			log.debug(f"processing all files in folder")
			# for file in ["TinyStories_all_data/data00.json"]:
			for file in sorted(glob.glob(os.path.join(file, "*"))):
				yield from gather_files(file)
		elif os.path.isfile(file):
			log.debug(f"assume file {file} is a pointer to a file")
			with open(file, 'r') as o:
				yield from process_json_dict(json.load(o))
		else:
			log.error(f"unsupported filetype {file}, of type {type(file)}")
			raise Exception 
	else:
		log.error(f"unsupported datatype {file}, of type {type(file)}")
		raise Exception 
	

def process_json_dict(file:dict):
	if not isinstance(file, list):
		log.error(f"unsupported datatype file must be of type list, found {type(file)}")
		raise Exception 
	for e in file:
		if not isinstance(e, dict):
			log.error(f"unsupported datatype entry must be of type dict, found {type(e)}")
			raise Exception 
		yield e

def kget(d:dict, key:str):
	if isinstance(d, dict):
		# print(d, key, type(key))
		if '.' in key:
			keys = key.split('.')
			return kget(d[keys[0]], '.'.join(keys[1:]))
		else:
			return d[key]
	else:
		log.error(f"d must be of type dict, found {type(d)}")
		raise Exception 

def kset(d, key:str, v:Any):
	if isinstance(d, dict):
		if '.' in key:
			keys = key.split('.')
			return kset(d[keys[0]], '.'.join(keys[1:]))
		else:
			d[key] = v
			return d
	else:
		log.error(f"d must be of type dict, found {type(d)}")
		raise Exception 

# 80000000 = 80MB
def write_outputs(file, split_size_bytes = 80000000, overwrite=False):
	""" Generator type writing pipline. used via gen.send(data). Calls str() around data."""
	if isinstance(file, str):
		if os.path.isdir(file):
			log.debug(f"assume file {file} is a folder")
			log.debug(f"putting files in folder with enumeration")
			counter = 1
			# create folder/test001.txt files
			partfile = os.path.join(file, "part" + str(counter) + ".txt")
			if os.path.isfile(partfile):
				if not overwrite:
					log.error(f"output file present, overwrite: {overwrite}, stopping")
					raise Exception("File already present")
				else:
					log.warning(f"output file present, overwrite: {overwrite}, deleting old file")
					os.remove(partfile)
			while True:
				data = yield
				if data == None:
					continue
				
				with open(partfile, 'a') as o:
					o.write(str(data))
				if os.path.getsize(partfile) > split_size_bytes:
					counter = counter + 1
					partfile = os.path.join(file, "part" + str(counter) + ".txt")
					if os.path.isfile(partfile):
						if not overwrite:
							log.error(f"output file present, overwrite: {overwrite}, stopping")
							raise Exception("File already present")
						else:
							log.warning(f"output file present, overwrite: {overwrite}, deleting old file")
							os.remove(partfile)
			#print("exit generator")
		elif os.path.isdir(os.path.dirname(file)):
			log.debug(f"assume file {file} is a pointer to a file")
			if os.path.isfile(file) and not overwrite:
				log.error(f"output file present, overwrite: {overwrite}, stopping")
				raise Exception("File already present")
			else:
				log.warning("Overwriting file")
			with open(file, 'w') as o:
				while True:
					data = yield
					o.write(str(data))
			#print("exit generator")
		else:
			log.error(f"unsupported filetype {file}, of type {type(file)}")
			raise Exception(f"unsupported filetype {file}")

def clean_punctuation(s:str) -> None:
	s = s.strip()
	s = s.replace('\\', '')
	s = s.replace('  ', ' ')
	s = s.replace('–', '-')
	s = s.replace(' — ', ' - ')
	s = s.replace('—', ' - ')
	s = s.replace('…', '...')
	s = s.replace('“', '"')
	s = s.replace('”', '"')
	s = s.replace('’', '\'')
	return s

def check_ascii(s:str) -> bool|None:
	for c in s:
		if ord(c) != 10 and (ord(c) > 127 or ord(c) < 32):
			# print("offending char:", ord(c), c)
			return True

# ` is usually as strange punctuation, or `` ''
# / is for Tom/Lily stories, "he/she"
# * is usually in *emphasis*, but often incorrect, or in *** separating parts of a story
# $ is usually correct
# & is as an abbrev, or &rsquo;s
# ~ is rare and not used well
# # is wrong, hashtags and rarely as numbers
# there's one mistake for % and the rest are ok. I decided to clean it anyway
# [ is used wrong
# _ is used poorly
# = is sometimes used for addition but more often for mistakes
# + is for addition, abbreviation, and A+. but has some mistakes
# ( ) is about 80% correctly used
_illegal_chars = ['|', '<',  '/' , '`', '*', '=', '_', '&', '@', '~', '#', '%' , '%' , '[', ']', '+', '(', ')']
def check_odd_symbols(s:str) -> bool|None:
	for c in s:
		if c in _illegal_chars:
			return True

def check_len(s:str) -> bool|None:
	if len(s) < 50 or len(s) == 0:
		return True
	
def check_punct(s:str) -> bool|None:
	if s[-1] != '.' and s[-1] != '!' and s[-1] != '"' and s[-1] != '?':
		return True
	

if __name__ == "__main__": 
	run_standalone()
	pass