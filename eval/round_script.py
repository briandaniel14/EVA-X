
import json
import re
import sys

def round_floats_in_obj(obj, ndigits=3):
	if isinstance(obj, float):
		return round(obj, ndigits)
	elif isinstance(obj, list):
		return [round_floats_in_obj(x, ndigits) for x in obj]
	elif isinstance(obj, dict):
		return {k: round_floats_in_obj(v, ndigits) for k, v in obj.items()}
	else:
		return obj

def process_log_file(input_path, output_path=None):
	with open(input_path, 'r') as f:
		lines = f.readlines()
	new_lines = []
	json_pattern = re.compile(r'^\{.*\}$')
	for line in lines:
		line_strip = line.strip()
		if json_pattern.match(line_strip):
			try:
				obj = json.loads(line_strip)
				obj_rounded = round_floats_in_obj(obj)
				new_lines.append(json.dumps(obj_rounded, ensure_ascii=False) + '\n')
			except Exception:
				new_lines.append(line)
		else:
			# Also round floats in lines like 'max_auc=0.806611575942099'
			float_pattern = re.compile(r'([-+]?[0-9]*\.[0-9]+)')
			def round_match(m):
				return str(round(float(m.group(0)), 3))
			new_line = float_pattern.sub(round_match, line)
			new_lines.append(new_line)
	with open(output_path or input_path, 'w') as f:
		f.writelines(new_lines)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python round_script.py <log_file> [output_file]")
		sys.exit(1)
	input_path = sys.argv[1]
	output_path = sys.argv[2] if len(sys.argv) > 2 else None
	process_log_file(input_path, output_path)
