# SureInspect frontend (Person 3)

React + Vite + Tailwind v4. Lives in `frontend/` of the monorepo.

```bash
git switch -c feat/frontend-scaffold
npm install                 # commit package-lock.json (CI requires it)
npm run gen:types           # writes src/services/api-types.ts from ../contracts/openapi.json
npm run dev                 # proxies /api to http://127.0.0.1:8000
npm run build
```

Notes
- `src/types.ts` is a thin view-model. After `gen:types`, switch it to aliases of `components["schemas"]` so there is no parallel hand-written API type.
- Confirm against the generated types: the `/api/categories` and `/api/samples` field names, and the three simulation form fields in `services/api.ts` (`simulate_blur`, `simulate_glare`, `simulate_lighting_shift`).
- Assets are fetched with the bearer header and turned into object URLs (revoked on cleanup). No secrets in `VITE_*`.
- Token lives in `sessionStorage`; one transparent renewal on 401.
