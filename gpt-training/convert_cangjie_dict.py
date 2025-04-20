"""
Takes the original RIME Wubi86 character to keystroke mapping and converts
it into a character to component mapping, writing it to a new yaml file.
"""
key_to_component = {
    'a': '日',
    'b': '月',
    'c': '金',
    'd': '木',
    'e': '水',
    'f': '火',
    'g': '土',
    'h': '竹',
    'i': '戈',
    'j': '十',
    'k': '大',
    'l': '中',
    'm': '一',
    'n': '弓',
    'o': '人',
    'p': '心',
    'q': '手',
    'r': '口',
    's': '尸',
    't': '廿',
    'u': '山',
    'v': '女',
    'w': '田',
    'x': '難',
    'y': '卜',
    'z': '重',
    "'": "'"
}


def convert_cangjie_to_component_yaml(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        parsing = False

        for line in fin:
            stripped = line.strip()

            # Preserve header, comments, empty lines
            if not parsing:
                fout.write(line)
                if stripped == "...":
                    parsing = True
                continue
            if not stripped or stripped.startswith('#'):
                fout.write(line)
                continue

            def convert_code(code):
                s = ""
                for k in code:
                    if k not in key_to_component:
                        print(f"Key {k} not found in key_to_component")
                    s += key_to_component.get(k, f"[{k}]")
                return s
                return ''.join(key_to_component.get(k, f"[{k}]") for k in code)
            
            parts = stripped.split('\t')
            if len(parts) == 3:
                char, code, alt = parts
                new_code = convert_code(code)
                new_alt = convert_code(alt)
                fout.write(f"{char}\t{new_code}\t{new_alt}\n")
            elif len(parts) == 2:
                char, code = parts
                new_code = convert_code(code)
                fout.write(f"{char}\t{new_code}\n")
            else:
                print(f"Malformed line: {line}")
                fout.write(line)  # Leave malformed lines untouched

    print("Done")


if __name__ == "__main__":
    input_path = "/mnt/storage/ntunggal/cangjie5.dict.yaml"
    output_path = "cangjie5-components.dict.yaml"
    convert_cangjie_to_component_yaml(input_path, output_path)