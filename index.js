(n => {
    "use strict";
    let t, r, e, i, u = (() => {
            let n = "abcdefghijklmnopqrstuvwxyz'",
                t = n.length,
                r = {};
            n.split("").forEach(((n, t) => r[n] = t));
            let e = (n, t) => {
                    for (let [r, e] of t) n = n.replace(r, e);
                    return n
                },
                i = (n, i, u) => {
                    if (n.length < 2) return "'" !== n;
                    n = e(n, i);
                    let f = n.length;
                    if (!f || f > 14 || /0/.test(n)) return !1;
                    let o, c = 0,
                        a = 1,
                        h = 0,
                        l = 0;
                    for (o = 0; f - 1 > o; o++) {
                        let e = u[(r[n[o]] + 1) * t + r[n[o + 1]]];
                        if (e > l && (l = e), !e) return !1;
                        if (4 > e && h++, 12 > e) {
                            if (c++, c > 2) return !1
                        } else c = 0;
                        a *= e / 255
                    }
                    if (4 > l || h / f > .4 || f > 3 && (2e-7 > a || a > .95)) return !1;
                    let s = (n.match(/[aeiouy]/g) || []).length;
                    return !(f > 3 && (f / 10 > s || s >= f / 1.1)) && 0
                },
                u = (n, e, i) => {
                    return n.length > 3 && e[(r[n[0]] + 1) * t + r[n[1]]] < 255 && !i.a(n.substr(0, 3))
                };
            return {
                b: e,
                c: i,
                d: u,
                e: n,
                f: r
            }
        })(),
        f = (() => {
            let n = function (n) {
                let t = this;
                t.g = new Uint8Array(n), t.a = (n => {
                    let r = t.h(n),
                        e = r % 8,
                        i = Math.floor(r / 8) % t.g.length;
                    return 0 !== (t.g[i] & 1 << e)
                }), t.h = (n => {
                    var t = 1;
                    for (let r of n) t = 49 * t + r.charCodeAt(0) & 4294967295;
                    return Math.abs(t)
                })
            };
            return n
        })();
    n.init = (n => {
        let u = 0,
            o = 16,
            c = t => (t ? n : n.buffer).slice(o, o += n.readUInt32LE(4 * u++));
        t = new f(c()), e = new f(c()), r = new Uint8Array(c()), i = c(!0).toString().split("\n").map((n => {
            let t = n.split("\t");
            return [new RegExp(t[0]), t[1] || ""]
        }))
    }), n.test = (n => {
        let f = u.c(n, i, r);
        if (0 !== f) return f;
        let o = u.b(n, i);
        return !u.d(o, r, e) && t.a(o)
    })
})(module.exports);