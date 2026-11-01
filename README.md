# IFK-TDK 2026: Vasúti Infrastruktúra és Ingatlanpiaci Áralakulás

## A közlekedésorientált városfejlesztés ingatlanpiaci sajátosságai vasúti környezetben
**Empirikus térökonometriai elemzés és ingatlanfejlesztési döntéstámogatás a budapesti Kőbánya és a bécsi Leopoldstadt példáján**

---

### 🌐 Élő Kutatási Dashboard (GitHub Pages)

A teljes interaktív kutatási eredménytár közvetlenül elérhető böngészőből GitHub Pages-en keresztül:

👉 **[Kutatási Dashboard Megnyitása (`index.html`)](index.html)**

* **MapLibre GL 3D vektoros térkép:** Budapest X. Kőbánya ($N = 269$) és Bécs II. Leopoldstadt ($N = 378$) geokódolt tranzakciós pontjai.
* **Valós idejű gyalogos útvonaltervezés:** OSRM útvonalkereső az állomási bejáratokig.
* **24 db publikációs minőségű Plotly diagram:** 5 tematikus fülbe rendezve (Piacmorfológia, Hedonikus OLS, Térökonometria & GWR, Térinformatika, Gépi Tanulás & LVC).
* **100%-ban önálló és kliensoldali:** Nem igényel háttérszervert vagy adatbázist.

---

### 📁 A Repository Tartalma

Ez a repository a kutatás reprodukálható számítási motorját és az interaktív webes felületét tartalmazza:

| Fájl | Típus | Leírás |
| :--- | :--- | :--- |
| **[`index.html`](index.html)** | Webes Dashboard | Az önálló, egyetlen állományból futó interaktív GitHub Pages kutatási felület. |
| **[`Számítások.ipynb`](Számítások.ipynb)** | Jupyter Notebook | **Master ökonometriai és ML számítási motor:** 68 cella, M1–M4 OLS modellek (HC3 SE), robusztussági diagnosztika (VIF, BP, White, RESET, Jarque-Bera), térökonometria (Moran-I, SAR, SEM, LeSage & Pace 46,6% spillover), GWR és XGBoost/CatBoost SHAP TreeExplainer. |
| **[`Vizualizacio.ipynb`](Vizualizacio.ipynb)** | Jupyter Notebook | **Master térinformatikai és vizualizációs motor:** 25 cella, a 24 db Plotly ábra és a webes dashboard generálása. |
| **`.nojekyll`** | Konfiguráció | Jekyll feldolgozás mellőzése a statikus GitHub Pages azonnali kiszolgálásához. |
| **`.github/workflows/deploy-pages.yml`** | CI/CD | Automatikus GitHub Pages publikálás push eseménykor. |

---

### 🚀 GitHub Pages Beállítás

1. Menj a GitHub repository felületén: **Settings** → **Pages**
2. **Build and deployment** résznél válaszd:
   * **Source:** `Deploy from a branch`
   * **Branch:** `main` (vagy `master`) / `/ (root)`
   * Kattints a **Save** gombra.
3. 1 percen belül az oldal elérhető a `https://<felhasznalonev>.github.io/<repo-nev>/` címen.
