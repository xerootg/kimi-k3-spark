#!/usr/bin/env python3
"""Draft second-pass ledger rows from Spark outputs (standard library only).

Usage: python3 tools/glm53/window_digest.py FILE [FILE ...]
Files are independent captures; overlapping runner/leg captures are not merged.
Missing or clipped evidence is never treated as a successful correctness gate.
"""
import argparse
from pathlib import Path
import re

GOLD = '9449ae5738'
NUMBER = r'\d+(?:\.\d+)?'
STAMP = re.compile(r'^\s*\d+(?:\.\d+){2,}\s+[A-Z]\s+')
RUN = re.compile(r'^(\S+)\s+r([012])\s+(.*)')


def clean(line):
    line = line.strip()
    while STAMP.match(line):
        line = STAMP.sub('', line)
    return line


def windows(text, filename):
    """Split explicit headers, otherwise infer the window from DONE/name."""
    chunks = []
    label = None
    lines = []
    for raw in text.splitlines():
        line = clean(raw)
        header = re.match(r'===\s+window\s+(\S+)(.*)', line)
        if header:
            if label is not None or lines:
                chunks.append((label, lines))
            label = header[1] + (f' ({header[2].strip()})' if header[2].strip() else '')
            lines = []
        else:
            lines.append(line)
    if label is not None or lines:
        chunks.append((label, lines))
    for label, lines in chunks:
        if label is None:
            done = re.search(r'WIN\d*([A-Z]+)-DONE', '\n'.join(lines))
            label = done[1].lower() if done else {'pi': 'p', 'pol': 'r'}.get(
                Path(filename).stem.removeprefix('legs-'), Path(filename).stem.removeprefix('legs-'))
        yield label, lines


def render(label, lines, filename):
    legs = {}
    details = []
    hashes = []
    other_hashes = []
    current = None
    # Hook-only summaries at the end of u cannot be assigned to the last leg.
    hook_only = not any('real bucket:' in line for line in lines)
    for index, line in enumerate(lines):
        if re.match(r'^\S+ MemAvailable', line):
            current = None
        run = RUN.match(line)
        long = re.match(r'^(\S+)\s+\d+-token prompt:', line)
        config = re.match(r'^(\S+)\s+(?:LFU=|remote deadline_us=)', line)
        section = re.match(r'^==\s+(\S+)$', line)
        if run or long or config or section:
            current = (run or long or config or section)[1]
            legs.setdefault(current, {'runs': {}, 'long': [], 'fill': None, 'ubatch': None})
        if run:
            tail = run[3]
            speed = re.search(r'(?:^|decode=)(' + NUMBER + r')\s+tok/s', tail)
            if speed:
                legs[current]['runs'][run[2]] = speed[1]
            elif 'pp=' in tail:
                legs[current]['long'].append('r' + run[2] + ' ' + tail)
            if 'prompt_n=' not in tail:
                digest = re.search(r'\bmd5[=\s]+([0-9a-fA-F]+)', tail)
                hashes.append(digest[1].lower() if digest else None)
        if long:
            legs[current]['long'].append(line.split(':', 1)[1].strip())
        if not run and not long and not line.startswith('cache-ram='):
            for value in re.findall(r'\bmd5[=\s]+([0-9a-fA-F]+)', line):
                if value not in other_hashes:
                    other_hashes.append(value)
        if line.startswith('cache-ram='):
            details.append(line)
        if 'fill_late' in line and re.search(r'\d+ lookups, hit rate', line):
            # Failed legs may only be identified by a trailing MemAvailable line.
            owner = current
            for future in lines[index + 1:]:
                mem = re.match(r'^(\S+) MemAvailable', future)
                if mem:
                    owner = mem[1]
                    break
                if RUN.match(future) or future.startswith('==='):
                    break
            # A decreasing counter without a leg marker is an unlabeled server.
            if owner in legs and legs[owner]['fill']:
                old = int(re.search(r'(\d+) lookups', legs[owner]['fill'])[1])
                new = int(re.search(r'(\d+) lookups', line)[1])
                if new < old:
                    owner = None
                    current = None
            if owner:
                legs.setdefault(owner, {'runs': {}, 'long': [], 'fill': None, 'ubatch': None})['fill'] = line
            else:
                details.append('unlabeled ' + line)
        if re.search(r'ubatch\s*>\s*8:', line):
            if current:
                legs[current]['ubatch'] = re.split(r'ubatch\s*>\s*8:', line, maxsplit=1)[1].strip()
            else:
                details.append(line)
        if 'host bank hooks:' in line or 'real bucket:' in line:
            owner = 'unlabeled' if hook_only else (current or 'unlabeled')
            summary = line[line.index('host bank'):]
            real = re.search(r'\breal=\d+/(' + NUMBER + r')ms', line)
            if real:
                summary += f'; real-hook {float(real[1]) / 384:.2f} ms/token (total / (3 x 128), includes prompt hooks)'
            details.append(owner + ' ' + summary)
        elif re.search(r'\b[\w-]*build[\w-]*=\d+', line) or 'ABORT-WINDOW' in line:
            details.append(line)
        elif any(key in line for key in ('FAST_EXIT=', 'GATE-FAIL', 'JSONDecodeError:', ' died:',
                                          'ORDER-CHECK', 'LAYOUT-INDEPENDENT=', 'VARIES=', 'PP stock=',
                                          'precision=', 'place=', 'SUMMARY ', 'MISMATCH')):
            details.append(line)
    parts = []
    for name, leg in legs.items():
        runs = leg['runs']
        if runs:
            parts.append(f"{name} r1/r2 {runs.get('1', '?')}/{runs.get('2', '?')} tok/s"
                         + (f" (r0 {runs['0']})" if '0' in runs else ''))
        parts.extend(name + ' ' + item for item in leg['long'])
    if hashes:
        good = sum(h is not None and h.startswith(GOLD) for h in hashes)
        parts.append(f'md5 gold `{GOLD}` in all {len(hashes)} observed short-prompt passes' if good == len(hashes)
                     else f'md5 gold {good}/{len(hashes)} short-prompt passes; mismatch/missing: '
                     + ', '.join(str(h or 'missing') for h in hashes if h is None or not h.startswith(GOLD)))
    else:
        parts.append('short-prompt md5: not in the window output')
    for name, leg in legs.items():
        if leg['fill']:
            parts.append(name + ' ' + leg['fill'])
        if leg['ubatch']:
            parts.append(name + ' ubatch > 8: ' + leg['ubatch'])
    if legs and not any(leg['fill'] for leg in legs.values()):
        parts.append('decode `fill_late ... hit rate`: not in the window output')
    for name in legs:
        if not name.endswith('0'):
            continue
        prefix = name[:-1]
        quartet = [prefix + suffix for suffix in ('0', '1', '0b', '1b')]
        if all(n in legs and all(r in legs[n]['runs'] for r in ('1', '2')) for n in quartet):
            off, on = [sum(float(legs[prefix + suffix]['runs'][r])
                           for suffix in suffixes for r in ('1', '2')) / 4
                       for suffixes in (('0', '0b'), ('1', '1b'))]
            if off:
                parts.append(f'{prefix} four-pass means: off {off:.3f}, on {on:.3f} tok/s; gain {100 * (on / off - 1):+.1f} %')
    if other_hashes:
        parts.append('other md5s seen (not the short-prompt gold gate): ' + ', '.join(other_hashes))
    parts.extend(details)
    parts.append('**verdict**')
    cells = [label, f'draft ({Path(filename).name})', '. '.join(parts)]
    return '| ' + ' | '.join(cell.replace('|', '&#124;') for cell in cells) + ' |'


def digest(text, filename):
    return '\n'.join(render(label, lines, filename) for label, lines in windows(text, filename))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.files:
        try:
            print(digest(path.read_text(encoding='utf-8', errors='replace'), path.name))
        except OSError as error:
            parser.exit(1, f'{path}: {error}\n')


if __name__ == '__main__':
    main()
