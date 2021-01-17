MathJax.Hub.Config({
    TeX: {
        Macros:
        {
            bs: ["{\\boldsymbol #1}", 1],
            d: ["{\\mathrm{d}#1}",1],
            dint: ["{\\,\\mathrm{d}#1}", 1],
            subindex: [" #1\_\{\\mathrm \{ #2 \} \}", 2],

            gpi: ["\\text\{ \\\greektext p\}\}"],
            gtheta: ["\\text \{ \\\greektext j \}"],
            gmu: ["\\text \{ \\greektext m \}"],
            geta: ["\\text \{ \\greektext h \}"],
            gLambda: ["\\text \{ \\greektext L\}"],
            laplace: ["\\text \{ \\greektext D \}"],

            ex: ["\\boldsymbol \{ e \}\_x"],
            ey: ["\\boldsymbol \{ e \}\_y"],
            ez: ["\\boldsymbol \{ e \}\_z"],

            etheta: ["\\boldsymbol \{ e \}\_\\theta"],
            ephi: ["\\boldsymbol \{ e \}\_\\varphi"],

            Reynolds: ["\\mathit \{ Re\}"],
            magReynolds: ["\\mathit \{ Rm\}"],
            Prandtl: ["\\mathit \{ Pr\}"],
            magPrandtl: ["\\mathit \{ Pm\}"],
            Rayleigh: ["\\mathit \{ Ra\}"],
            modRayleigh: ["\\mathit \{ Ra\}\^*"],
            Rossby: ["\\mathit \{ Ro\}"],
            Elsasser: ["\\mathit \{ \\text \{ \\greektext L\} \}"],
            Euler: ["\\mathit \{ Eu\}"],
            Froude: ["\\mathit \{ Fr\}"],
            Ekman: ["\\mathit \{ Ro\}"],
            Strouhal: ["\\mathit \{ St\}"],

            p: ["\\partial"],
            dd: ["\\frac{\\mathrm{d} #1 \} \{ \\mathrm{d} #2 \}", 2],
            ddsqr: ["\\frac{\\mathrm{d}^2 #1\} \{ \\mathrm{d} #2^2 \}", 2],
            pd: ["\\frac{\\p #1 \} \{\\p #2\}", 2],
            ppd: ["\\frac{\\p^2 #1\} \{\\p #2 \\p #3 \}", 3],
            pdsqr: ["\\frac{\\p^2 #1\} \{ \\p #2^2 \}", 2],

            cdott: ["\\, \{\\cdot\} \{ \\cdot \}\\,"]
        }
    }
});

