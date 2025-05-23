import json
import importlib.resources

from typing import List
from dataclasses import dataclass, field


def available_sets() -> List[str]:
    """List available gene sets in juzi.mg"""
    return [
        "CancerBreast",
        "CancerPathways",
        "CellCycle"
    ]


@dataclass
class CancerBreast:
    """Breast cancer gene markers spanning subtypes and drivers

    Notes
    -----
    The subtype specific markers were taken as the top 10 ranking genes
    from the PAM50 centroids (i.e. genes with highest subtype expression) the
    curated sets are smaller sets known to be expressed in certain subtypes.
    The TNBC markers from Burstein et al. were collected from Table 2 which
    outlined the genes with significant overexpression.

    References
    ----------
    .. [1] J.S. Parker et al. Supervised Risk Predictor of Breast Cancer Based
       on Intrinsic Subtypes. Journal of Clinical Oncology. 2009.
    .. [2] M.D Burstein et al. Comprehensive genomic analysis identifies novel 
       subtypes and targets of triple-negative breast cancer.
       Clinical Cancer Research. 2015.
    """

    PAM50: List[str] = field(init=False)
    PAM50_BASAL: List[str] = field(init=False)
    PAM50_HER2: List[str] = field(init=False)
    PAM50_LUMA: List[str] = field(init=False)
    PAM50_LUMB: List[str] = field(init=False)
    PAM50_NORMAL: List[str] = field(init=False)

    CURATED_BASAL: List[str] = field(init=False)
    CURATED_HER2: List[str] = field(init=False)
    CURATED_LUMA: List[str] = field(init=False)
    CURATED_LUMB: List[str] = field(init=False)

    TNBC_BURSTEIN_LAR: List[str] = field(init=False)
    TNBC_BURSTEIN_MES: List[str] = field(init=False)
    TNBC_BURSTEIN_BLIS: List[str] = field(init=False)
    TNBC_BURSTEIN_BLIA: List[str] = field(init=False)

    def info(self):
        return self.__annotations__

    def all(self):
        markers = []
        for key in self.info().keys():
            markers += getattr(self, key)
        return markers

    def __post_init__(self):
        with importlib.resources.open_text(
            "juzi.mg",
            "cancer_breast.json"
        ) as f:
            data = json.load(f)

        self.PAM50 = data["PAM50"]
        self.PAM50_BASAL = data["PAM50_BASAL"]
        self.PAM50_HER2 = data["PAM50_HER2"]
        self.PAM50_LUMA = data["PAM50_LUMA"]
        self.PAM50_LUMB = data["PAM50_LUMB"]
        self.PAM50_NORMAL = data["PAM50_NORMAL"]

        self.CURATED_BASAL = data["CURATED_BASAL"]
        self.CURATED_HER2 = data["CURATED_HER2"]
        self.CURATED_LUMA = data["CURATED_LUMA"]
        self.CURATED_LUMB = data["CURATED_LUMB"]

        self.TNBC_BURSTEIN_LAR = data["TNBC_BURSTEIN_LAR"]
        self.TNBC_BURSTEIN_MES = data["TNBC_BURSTEIN_MES"]
        self.TNBC_BURSTEIN_BLIS = data["TNBC_BURSTEIN_BLIS"]
        self.TNBC_BURSTEIN_BLIA = data["TNBC_BURSTEIN_BLIA"]


@dataclass
class CancerPathways:
    """Genes from various canonical cancer pathways

    Notes
    -----
    The splicesome and JAK-STAT markers were collected from the
    KEGG database. The remaining from Sanchez-Vega et al. 2018.

    References
    ----------
    .. [1] F. Sanchez-Vega et al. Oncogenic Signaling Pathways in The Cancer
       Genome Atlas. Cell. 2018.
    .. [2] M. Kanehisa et al. KEGG: Kyoto Encyclopedia of Genes and Genomes. 
       Nucleic Acids Research. 2000.
    """

    CELL_CYCLE: List[str] = field(init=False)
    HIPPO: List[str] = field(init=False)
    JAK_STAT: List[str] = field(init=False)
    MYC: List[str] = field(init=False)
    NOTCH: List[str] = field(init=False)
    PI3K: List[str] = field(init=False)
    RTK_RAS: List[str] = field(init=False)
    SPLICESOME: List[str] = field(init=False)
    TGF_BETA: List[str] = field(init=False)
    TP53: List[str] = field(init=False)
    WNT: List[str] = field(init=False)

    def info(self):
        return self.__annotations__

    def all(self):
        markers = []
        for key in self.info().keys():
            markers += getattr(self, key)
        return markers

    def __post_init__(self):
        with importlib.resources.open_text(
            "juzi.mg",
            "cancer_pathways.json"
        ) as f:
            data = json.load(f)

        self.CELL_CYCLE = data["CELL_CYCLE"]
        self.HIPPO = data["HIPPO"]
        self.JAK_STAT = data["JAK_STAT"]
        self.MYC = data["MYC"]
        self.NOTCH = data["NOTCH"]
        self.PI3K = data["PI3K"]
        self.RTK_RAS = data["RTK_RAS"]
        self.SPLICESOME = data["SPLICESOME"]
        self.TGF_BETA = data["TGF_BETA"]
        self.TP53 = data["TP53"]
        self.WNT = data["WNT"]


@dataclass
class CellCycle:
    """Cell cycle gene markers

    Notes
    -----
    The G1S and G2M markers were collected from Tirosh et al. 2016
    and the CDK/Cyclin genes were collected from HUGO.

    References
    ----------
    .. [1] I. Tirosh et al. Dissecting the multicellular ecosystem of
       metastatic melanoma by single-cell RNA-seq. Science. 2016.
    """

    G1S: List[str] = field(init=False)
    G2M: List[str] = field(init=False)
    CDK: List[str] = field(init=False)
    CYCLIN: List[str] = field(init=False)

    def info(self):
        return self.__annotations__

    def all(self):
        markers = []
        for key in self.info().keys():
            markers += getattr(self, key)
        return markers

    def __post_init__(self):
        with importlib.resources.open_text("juzi.mg", "cell_cycle.json") as f:
            data = json.load(f)

        self.G1S = data["G1S"]
        self.G2M = data["G2M"]
        self.CDK = data["CDK"]
        self.CYCLIN = data["CYCLIN"]
