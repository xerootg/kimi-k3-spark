#!/usr/bin/env python3
"""Generate survey-only sources; fail closed if source anchors change."""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
def replace(s, a, b):
    if s.count(a) != 1:
        raise RuntimeError('source anchor changed: ' + a[:80])
    return s.replace(a, b)
for name in ('g53x', 'g53c'):
    s = (root / 'tools/glm53' / (name + '.cpp')).read_text()
    anchor = 't.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());'
    raw = '{ double survey_us=std::chrono::duration<double, std::micro>(t1-t0).count(); t.push_back(survey_us); printf("SURVEY k=%d us=%.6f\\n", k, survey_us); fflush(stdout); }'
    s = replace(s, anchor, raw)
    if name == 'g53x':
        s = replace(s, '    // load: one read + repack per record, straight into host USM', '''    const bool survey_disk = getenv("K3X_DISK_TIER") && atoi(getenv("K3X_DISK_TIER"));
    // Survey prototype: exactly eight repacked slots, same layer only.
    if (survey_disk && want.size() > 8) want.resize(8);
    // load: one read + repack per record, straight into host USM''')
        s = replace(s, '    long zero_rows = 0; int nconn = 0;', '''    int survey_fd = -1; void * survey_buf = nullptr;
    size_t survey_bytes = 0;
    for (auto & r : store->records) survey_bytes = std::max(survey_bytes, (size_t)r.bytes);
    if (survey_disk) {
        survey_fd = open(storep, O_RDONLY | O_DIRECT);
        if (survey_fd < 0 || posix_memalign(&survey_buf, 4096, survey_bytes)) return 4;
        if (map.size() != 8) return 4;
    }
    std::unordered_map<long, unsigned long> survey_age;
    unsigned long survey_tick = 0;
    long zero_rows = 0; int nconn = 0;''')
        s = replace(s, '            int ki = -1; bool ok = true;', '''            int survey_reads = 0;
            if (survey_disk) for (int i = 0; i < k; ++i) {
                long gid = hdr[1+i];
                const auto * r = store->find(gid);
                if (!r || layer_kind[gid/NX] < 0) continue;
                if (!map.count(gid)) {
                    auto victim = map.end();
                    for (auto it = map.begin(); it != map.end(); ++it) {
                        bool pinned = false;
                        for (int j=0;j<k;++j) if (hdr[1+j] == it->first) pinned = true;
                        if (!pinned && (victim == map.end() || survey_age[it->first] < survey_age[victim->first])) victim = it;
                    }
                    if (victim == map.end()) return 4;
                    auto slot = victim->second; survey_age.erase(victim->first); map.erase(victim);
                    ssize_t nr;
                    do { nr = pread(survey_fd, survey_buf, r->bytes, r->offset); } while(nr < 0 && errno == EINTR);
                    if (nr != (ssize_t)r->bytes) return 4;
                    slot.kind = layer_kind[gid/NX];
                    const auto & K = kinds[slot.kind];
                    auto * src = (uint8_t *)survey_buf;
                    auto * dst = const_cast<uint8_t *>(slot.p);
                    // Slots have L3's shape; reject any other kind before writing.
                    if (K.gsz + K.usz + K.dsz != 12091392) return 4;
                    repack(K.tg, src, dst, K.F, K.D);
                    repack(K.tu, src+K.gsz, dst+K.gsz, K.F, K.D);
                    repack(K.td, src+K.gsz+K.usz, dst+K.gsz+K.usz, K.D, K.F);
                    map[gid] = slot; ++survey_reads;
                }
                survey_age[gid] = ++survey_tick;
            }
            int ki = -1; bool ok = true;''')
        s = replace(s, '            if (!okw) break;', '            fprintf(stderr, "SURVEY_DISK k=%d reads=%d\\n", k, survey_reads);\n            fflush(stderr);\n            if (!okw) break;')
    (out / (name + '.cpp')).write_text(s)
# Header includes remain local to generated source.
for p in (root / 'tools/glm53').glob('*.inc'):
    (out / p.name).write_bytes(p.read_bytes())
for name in ('g53exact.h',):
    (out / name).write_bytes((root / 'tools/glm53' / name).read_bytes())
