import logging
log = logging.getLogger(__name__)

import re
import enum
import json
import glob
import os
from typing import Any

import enchant
import qtransform
args = [ "run=script", "run.file=none" ]
@qtransform.with_config(args, logging.DEBUG)
def run_standalone(cfg):
	log.info(f"running {__file__}")
	file = "/home/kuhmichel/.qtransform/datasets/downloads/TinyStories_all_data"
	stats = {
		"errors": 0,
		"succcess": 0,
	}
	words = dict()
	#pat = re.compile(r"(.+?)[\s'\",!,\.,\?,-]{1,}")
	pat = re.compile(r"([\w][\w]*'?\w?)")
	#english = enchant.Dict("en_US")
	english = enchant.DictWithPWL("en_US","mywords.txt")
	for _i, e in enumerate(gather_files(file)):
		yes, data = process_story(e)
		if not yes:
			stats['errors'] = stats['errors'] + 1
		else:
			stats['succcess'] = stats['succcess'] + 1
		if yes:
			story = data[Keys.story]
			
			story_words = pat.findall(story)
			for word in story_words:
				if not english.check(word):
					print(word)
					print(story)
		if _i > 100:
			break

class Keys(str, enum.Enum):
	story = 'story'
	instruction = 'instruction'
	source = 'source'
	instruction_prompt = 'instruction.prompt:'
	instruction_words = 'instruction.words'
	instruction_features = 'instruction.features'
	summary = 'summary'
	def __str__(self):
		return f'{str(self.value)}'


def process_story(data:dict):
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
			for file in glob.glob(os.path.join(file, "*")):
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

def write_outputs(file, split_size_bytes = 100000000, overwrite=False):
	""" Generator type writing pipline. used via gen.send(data). Calls str() around data."""
	if isinstance(file, str):
		if os.path.isdir(file):
			log.debug(f"assume file {file} is a folder")
			log.debug(f"putting files in folder with enumeration")
			counter = 1
			# create folder/test001.txt files
			partfile = os.path.join(file, "part" + str(counter) + ".txt")
			while True:
				data = yield
				# TODO
				with open(partfile, 'a') as o:
					o.write(str(data))
				if os.path.getsize(partfile) > split_size_bytes:
					counter = counter + 1
					partfile = os.path.join(file, "part" + str(counter) + ".txt")
		elif os.path.isdir(os.path.dirname(file)):
			log.debug(f"assume file {file} is a pointer to a file")
			if os.path.isfile(file) and not overwrite:
				log.error(f"output file present, overwrite: {overwrite}, stopping")
				raise Exception
			else:
				log.warning("Overwriting file")
			with open(file, 'w') as o:
				while True:
					data = yield
					o.write(str(data))
		else:
			log.error(f"unsupported filetype {file}, of type {type(file)}")
			raise Exception 

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