# ⚖️ Considerações éticas (reflexão para o relatório — Etapa 5)

## Restrição central
> É **vedado** treinar o modelo para imitar o estilo de **artistas vivos
> identificáveis** ou reproduzir **personagens e propriedade intelectual de
> terceiros** (Disney, Nintendo, etc.). Estilos históricos, tradições culturais
> e estéticas genéricas são **permitidos**; a imitação direcionada de um artista
> contemporâneo específico, **não**.

## Justificativa (texto da equipe para o relatório)
Esta restrição busca **proteger os direitos morais e patrimoniais dos
criadores**, evitar a **apropriação indevida de identidade artística**, reduzir
riscos de **violação de direitos autorais** e promover o **desenvolvimento
responsável da inteligência artificial**. Além disso, a medida contribui para a
**preservação da autoria humana**, da **originalidade criativa** e da
**transparência** na geração de conteúdo sintético, em conformidade com
princípios de **ética digital**, **propriedade intelectual** e **uso responsável
de tecnologias de IA**.

Por outro lado, é **permitida** a utilização de estilos históricos, movimentos
artísticos, tradições culturais, escolas estéticas, técnicas visuais e
características genéricas de linguagem visual que **não** estejam associadas à
identidade criativa exclusiva de um artista vivo específico.

## Como o projeto cumpre as restrições
1. **Estilo genérico/histórico**: trabalhamos com a estética **"pixel art 16
   bits"**, uma linguagem visual ampla e historicamente associada aos jogos dos
   anos 80/90 — não ao estilo pessoal de um artista vivo identificável.
2. **Sem personagens/IP protegidos**: o dataset não inclui mascotes ou
   personagens de franquias (Nintendo, Disney, etc.); apenas cenários e
   elementos genéricos (vilas, florestas, naves, cavernas...).
3. **Diversidade de autores**: coletamos **poucas imagens por artista** e
   buscamos **muitos artistas diferentes**, reduzindo o risco de o LoRA aprender
   e reproduzir o estilo de um único criador.
4. **Licenças compatíveis**: priorizamos itch.io (assets gratuitos), Pixabay e
   Pexels, com licenças que permitem o uso pretendido e sem restrição a treino
   de IA. Toda a origem está registrada em [`dataset/fontes.csv`](../dataset/fontes.csv).
5. **Transparência**: a UI declara explicitamente que o conteúdo é **gerado por
   IA**.
6. **Segurança e privacidade**: tokens/chaves ficam apenas nos **Secrets** do
   Hugging Face Spaces, nunca no código público.

## Consulta ao professor
A equipe registrou no fórum a abordagem (tema Pixel Art 16 bits, fontes itch.io,
licenças permitidas, diversidade de artistas) solicitando validação de
alinhamento com as restrições do trabalho.

## Limitações e riscos residuais
- Datasets pequenos podem **memorizar** amostras; mitigamos com diversidade,
  rank moderado e número de passos controlado.
- Licenças de assets "gratuitos" variam por autor; manter a coluna `license`
  de `fontes.csv` sempre atualizada e conferida.
