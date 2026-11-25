# Sistema de diseño de FoodSecurity AI V2 · «Atlas clínico»

Dirección: periodismo de datos editorial (papel cálido, titulares con serifa, cifras tabulares, filetes finos) cruzado con la precisión
clínica de un tablero de salud pública (verde-azulado de tinta, semántica sobria, sin decoración que no codifique datos).
Implementación: `app_streamlit/app/tema.py` (tokens, componentes, Plotly) + `app_streamlit/app/assets/estilos.css` (estructura y componentes).

## Tipografía
| Rol | Pila | Nota |
|---|---|---|
| Display (titulares, cifras KPI) | Iowan Old Style, Palatino Linotype, Palatino, Book Antiqua, Georgia | serifa del sistema |
| Texto | Segoe UI Variable Text, Segoe UI, system-ui, Helvetica Neue, Arial | sans del sistema |
| Mono (tablas, códigos) | Cascadia Mono, SF Mono, Consolas, Menlo | cifras tabulares |

Estrategia de fuentes: **pila de sistema afinada, sin incrustar fuentes**. No fue posible verificar sin conexión la licencia OFL ni conseguir un woff2
redistribuible, así que no se incluyó ninguna (cero red, cero licencias). Escala: 12 / 13 / 15 / 17 / 20 / 25,6 px y display fluido `clamp(1,85rem … 3,1rem)`.

## Tokens
- Espaciado (base 4 px): 4, 8, 12, 16, 24, 32, 48, 72. Radios: 6 / 12 / 20 px. Elevación: `--e1` (tarjeta), `--e2` (hover), `--e3` (héroe/acceso).
- Movimiento: 160 ms `cubic-bezier(.2,.7,.2,1)`; entradas `fs-rise` y esqueletos pulsantes solo bajo `prefers-reduced-motion: no-preference`; con `reduce` se anula toda animación y transición.
- Foco: anillo de 3 px ocre (`--fs-foco`) con `:focus-visible`; objetivos táctiles ≥ 44 px (botones, pestañas, píldoras, filas de navegación, entradas).
- Puntos de ruptura: 1200, 900, 640 y 420 px (el Inicio usa consultas de contenedor; ver «Responsividad»).
- Colores semánticos FIES (secuenciales por luminancia, forma + código en cada insignia): N0 verde-azulado, N1 ámbar, N2 naranja, N3 rojo ladrillo.
- Mapas: rampa secuencial `SEQ` (luminancia monótona) y divergente marrón↔verdeazulado `DIV` (anomalía de lluvia). Categórica Okabe-Ito.

## Tema (solo claro)
La plataforma tiene **una única paleta clara**: no hay selector de apariencia ni tokens oscuros. Los tokens (`tema.TOKENS`) se inyectan como variables CSS y
`.streamlit/config.toml` fija `[theme] base = "light"` con los mismos colores base, de modo que los widgets nativos de Streamlit tampoco siguen la
preferencia oscura del sistema o del navegador (verificado en Streamlit 1.41.1 con `prefers-color-scheme: dark` emulado: fondo, barra lateral, botones, selectores
y expanders siguen claros). Plantilla Plotly única: `fsai`.

## Responsividad
- Las rejillas del Inicio (KPIs, tira de estadísticos, ruta de 12 pasos, tablas) usan **consultas de contenedor** (`.fs-cq` + `@container`): reaccionan al ancho
  disponible, no al de la ventana (la barra lateral abierta lo reduce). KPIs 4→2→1 (si el número es impar, la última tarjeta ocupa el ancho completo: sin huérfanas);
  ruta 4→2→1 (≥ 900 px / ≥ 520 px / menos); tira 4→2; bandas del héroe `auto-fit`.
- Titular del héroe fluido con `clamp()`. Tablas: se desplazan dentro de su propio contenedor; con `apilar=True` (Inicio) pasan a tarjetas «encabezado · valor» en contenedores ≤ 600 px.
- Objetivos táctiles ≥ 44 px; con la barra lateral plegada o ≤ 900 px se reserva espacio superior para el botón de despliegue.
- Streamlit 1.41.1 pliega la barra lateral por defecto en móviles (`initial_sidebar_state="auto"`); en 768 px aparece abierta y empuja el contenido.
- Formato numérico español: `tema.num_es` (coma decimal; espacio fino sin separación como separador de miles desde 10 000, norma RAE/SI). Lo usan Inicio, `tabla_html` (columnas numéricas) y `comun.fmt_num/fmt_pct`.
- Acceso: el bloque (panel de marca + formulario) se centra vertical y horizontalmente (`min-height:100dvh`), ambos paneles con la misma altura; en móvil el panel de marca se compacta y se apila sobre el formulario.

## Componentes (`tema.py`)
`encabezado` (migas + riel de progreso de 4 fases + «Paso n de 12» + titular), `heroe`, `kpi_tarjetas` (delta con icono + texto, sparkline SVG con `<title>`),
`insignia_fies` / `insignia` (forma + texto), `tira_estadisticos`, `seccion`, `llamada` (ética/aviso/error/ok/info), `vacio` (con siguiente paso y enlace), `esqueleto`,
`tabla_html` (tabla común, escapa todo; vínculos solo `https://`), `pildoras` (radio horizontal con aspecto de píldora), `tarjeta_termino`, `ruta_analisis` (tubería de 12 pasos),
`mapa_coropletico` (contorno grueso para el quintil superior), `barras_ranking` (cifras escritas), iconos SVG en línea (`icono`), plantilla Plotly única, `num_es`.

## Accesibilidad
- Puntos de referencia semánticos: `nav` (migas), `ol` (riel y ruta), `section`/`article`/`aside` con `aria-label`, `role="status"`/`"note"`/`"alert"`, tablas con `caption`, `th scope="col"` y región desplazable con `tabindex`.
- SVG: decorativos con `aria-hidden`; gráficos mínimos con `role="img"`, `aria-label` y `<title>`.
- Nada se comunica solo con color: insignias con forma+código, deltas con icono+texto, mapas con contorno grueso del quintil superior y ranking con cifras como equivalente textual.
- Contraste (WCAG 2.x, calculado por `python app/tema.py`; mínimos: 4,5:1 texto, 3:1 no textual):

| Par (primer plano / fondo) | Uso | Ratio | Mínimo | Resultado |
|---|---|---|---|---|
| `tinta / fondo` (#0F1B24 / #F2EFE7) | Titulares sobre fondo | 15.20 | 4.5 | cumple |
| `texto / fondo` (#1D2A33 / #F2EFE7) | Texto base sobre fondo | 12.77 | 4.5 | cumple |
| `texto / superficie` (#1D2A33 / #FCFBF8) | Texto base sobre tarjeta | 14.18 | 4.5 | cumple |
| `suave / fondo` (#55616A / #F2EFE7) | Texto secundario sobre fondo | 5.53 | 4.5 | cumple |
| `suave / superficie` (#55616A / #FCFBF8) | Texto secundario sobre tarjeta | 6.14 | 4.5 | cumple |
| `suave / lateral` (#55616A / #EAE6DA) | Texto secundario en barra lateral | 5.09 | 4.5 | cumple |
| `acento-texto / superficie` (#0B5C5E / #FCFBF8) | Enlaces / acento como texto sobre tarjeta | 7.50 | 4.5 | cumple |
| `acento-texto / fondo` (#0B5C5E / #F2EFE7) | Enlaces / acento como texto sobre fondo | 6.75 | 4.5 | cumple |
| `sobre-acento / acento` (#FFFFFF / #0B5C5E) | Texto del botón primario | 7.76 | 4.5 | cumple |
| `heroe-texto / heroe-fondo` (#F3F1EA / #0C2F33) | Texto del héroe | 12.64 | 4.5 | cumple |
| `heroe-suave / heroe-fondo` (#B7CCC8 / #0C2F33) | Bajada del héroe | 8.49 | 4.5 | cumple |
| `heroe-ojo / heroe-fondo` (#8FD9C6 / #0C2F33) | Sobretítulo del héroe | 8.78 | 4.5 | cumple |
| `peligro-texto / superficie` (#9B1C16 / #FCFBF8) | Texto de error | 7.89 | 4.5 | cumple |
| `adv-texto / superficie` (#7A4600 / #FCFBF8) | Texto de advertencia | 7.51 | 4.5 | cumple |
| `ok-texto / superficie` (#17603E / #FCFBF8) | Texto de éxito | 7.30 | 4.5 | cumple |
| `info-texto / superficie` (#1B5287 / #FCFBF8) | Texto informativo | 7.80 | 4.5 | cumple |
| `n0-t / n0-f` (#14564A / #DCEEE9) | Insignia FIES N0 | 7.09 | 4.5 | cumple |
| `n1-t / n1-f` (#6B4A00 / #FAEFD2) | Insignia FIES N1 | 7.04 | 4.5 | cumple |
| `n2-t / n2-f` (#7A3607 / #FADFCB) | Insignia FIES N2 | 6.99 | 4.5 | cumple |
| `n3-t / n3-f` (#7F1D13 / #F6D9D4) | Insignia FIES N3 | 7.57 | 4.5 | cumple |
| `borde-ctl / superficie` (#7C7664 / #FCFBF8) | Borde de controles (no textual) | 4.38 | 3.0 | cumple |
| `foco / fondo` (#A64B0A / #F2EFE7) | Anillo de foco sobre fondo (no textual) | 5.03 | 3.0 | cumple |
| `foco / superficie` (#A64B0A / #FCFBF8) | Anillo de foco sobre tarjeta (no textual) | 5.59 | 3.0 | cumple |
| `n0 / superficie` (#3C8A7B / #FCFBF8) | Marca FIES N0 (no textual) | 3.97 | 3.0 | cumple |
| `n1 / superficie` (#B2801A / #FCFBF8) | Marca FIES N1 (no textual) | 3.38 | 3.0 | cumple |
| `n2 / superficie` (#C2601F / #FCFBF8) | Marca FIES N2 (no textual) | 4.07 | 3.0 | cumple |
| `n3 / superficie` (#A12B1E / #FCFBF8) | Marca FIES N3 (no textual) | 7.05 | 3.0 | cumple |

## Limitaciones conocidas de Streamlit 1.41.1
- Los selectores `data-testid` pueden cambiar entre versiones de Streamlit.
- `st.dataframe` dibuja en un lienzo que no se tematiza con CSS (usa los colores de `config.toml`); las tablas nuevas usan HTML propio.
- Los widgets nativos no aceptan clases: los estilos se aplican por `data-testid` y por contenedores con `key` (`st-key-…`).
- Los menús desplegables (portales de BaseWeb) se tematizan con selectores `[data-baseweb]` y pueden variar.
- El orden de foco sigue el orden del DOM de Streamlit; los enlaces «saltar al contenido» no son posibles sin JavaScript. El botón del riel y las migas no son interactivos.
- `st.navigation` recarga la página completa si se usa un `<a href>`; por eso los enlaces entre módulos usan `st.page_link`.
- Mapas: GeoJSON incluido con **licencia no verificada** (aviso visible en las páginas con mapa).

## Formulario de Inferencia y beeswarm
- El formulario se arma desde `fsai/catalogo_campos.py`: cada campo lleva pregunta corta visible, una línea de ayuda visible y detalle en el tooltip; secciones con 3 columnas que se apilan en el teléfono en el orden de lectura.
- Los cocientes (hacinamiento, dependencia económica, proporción del gasto en alimentos, asistencia escolar, puntaje FIES) se piden como datos simples y se calculan con la fórmula del entrenamiento.
- Interpretabilidad: el beeswarm se dibuja en la pestaña «Beeswarm por hogar» (muestra estratificada por nivel FIES, SHAP firmado, color = valor normalizado, categóricas en gris); donde no hay beeswarm almacenado se explica por qué y se ofrece calcularlo.

## Capturas
`docs/capturas/` (no incluidas en este repositorio) — login (360/768/1280/1920), inicio (360/414/768/1280/1440 y con menú abierto), dashboard, territorial y fuentes en claro a 1440 y 390 px (Chrome headless vía CDP; ver informe de la entrega).
