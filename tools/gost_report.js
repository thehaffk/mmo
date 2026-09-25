// Сборка отчёта по ГОСТ 7.32-2017 из упрощённого markdown.
// Запуск: node tools/gost_report.js <work>/report.md <work>/report.docx
//
// Формат исходника:
//   ---                              титульный лист
//   work: ЛАБОРАТОРНАЯ РАБОТА № 4
//   topic: Обработка многоцветных изображений
//   ---
//   # ВВЕДЕНИЕ                       заголовок 1 уровня (с новой страницы)
//   ## 1.1 Подраздел                 заголовок 2 уровня
//   абзац текста, **жирный**, `код`
//   - пункт / 1. пункт               перечисление а) б) / 1) 2)
//   ```python ... ```                фрагмент кода
//   ![Рисунок 1 — Подпись](figures/01.png)
//   Таблица 1 — Подпись               строка сразу перед markdown-таблицей
//   | a | b |
//
// Содержание со страницами: документ собирается дважды, между проходами
// LibreOffice рендерит его в PDF, а номера страниц заголовков берутся из pdftotext.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, Footer, PageNumber,
  Tab, TabStopType, LeaderType,
} = require('docx');

const MM = 56.7; // twip в миллиметре
const TEXT_WIDTH_MM = 210 - 30 - 15;
const FONT = 'Times New Roman';
const LOWER_LETTERS = 'абвгдежиклмнпрстуфхцшщэюя';

function parseSource(src) {
  const m = src.match(/^---\n([\s\S]*?)\n---\n/);
  const meta = {};
  if (m) {
    for (const line of m[1].split('\n')) {
      const i = line.indexOf(':');
      if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    }
  }
  return { meta, body: m ? src.slice(m[0].length) : src };
}

function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
    if (!part) continue;
    if (part.startsWith('**')) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith('`')) out.push(new TextRun({ ...base, text: part.slice(1, -1), font: 'Courier New', size: (base.size || 28) - 4 }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}

const bodyPara = (text, opts = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: Math.round(12.5 * MM) },
  spacing: { line: 360 },
  ...opts,
  children: runs(text),
});

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

function imageBlock(caption, file) {
  const { w, h, data } = pngSize(file);
  const maxW = TEXT_WIDTH_MM / 25.4 * 96; // px при 96 dpi
  const maxH = 200 / 25.4 * 96;
  const k = Math.min(maxW / w, maxH / h, 1);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120 },
      children: [new ImageRun({ type: 'png', data, transformation: { width: Math.round(w * k), height: Math.round(h * k) } })],
    }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: 360, after: 240 }, children: runs(caption) }),
  ];
}

function tableBlock(caption, rows) {
  const cells = rows.map(r => r.replace(/^\||\|$/g, '').split('|').map(c => c.trim()));
  const header = cells[0];
  const data = cells.slice(2); // вторая строка markdown-таблицы разделитель
  const total = Math.round(TEXT_WIDTH_MM * MM);
  // ширина колонки пропорциональна самому длинному тексту, но не меньше 24 мм,
  // чтобы заголовки вроде «Accuracy» не рвались посреди слова; недостающее забираем у самой широкой
  const longest = header.map((_, i) => Math.max(...[header, ...data].map(r => (r[i] || '').length)));
  const sum = longest.reduce((a, b) => a + b, 0);
  const minW = Math.min(Math.round(24 * MM), Math.floor(total / header.length));
  const widths = longest.map(n => Math.max(minW, Math.floor(total * n / sum)));
  const widest = widths.indexOf(Math.max(...widths));
  widths[widest] += total - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
  const mkRow = (vals, bold) => new TableRow({
    cantSplit: true,
    tableHeader: bold,
    children: vals.map((v, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      borders: { top: border, bottom: border, left: border, right: border },
      margins: { left: 80, right: 80 },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: runs(v, { size: 24, bold }) })],
    })),
  });
  const out = [];
  if (caption) out.push(new Paragraph({ keepNext: true, spacing: { before: 240, line: 360 }, children: runs(caption) }));
  out.push(new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [mkRow(header, true), ...data.map(r => mkRow(r, false))],
  }));
  out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

function titlePage(meta) {
  // на титуле одинарный интервал, иначе он не помещается на одну страницу
  const c = (text, opts = {}) => new Paragraph({
    alignment: AlignmentType.CENTER, ...opts,
    spacing: { line: 240, ...(opts.spacing || {}) },
    children: runs(text, { size: opts.size || 28, bold: opts.bold }),
  });
  const r = (text, before = 0) => new Paragraph({ indent: { left: Math.round(95 * MM) }, spacing: { line: 240, before }, children: runs(text) });
  return [
    c('ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ АВТОНОМНОЕ ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ'),
    c('МОСКОВСКИЙ ПОЛИТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ', { bold: true, spacing: { before: 120 } }),
    c('Факультет Информационных технологий', { spacing: { before: 360 } }),
    c('Кафедра Информатики и информационных технологий'),
    c('направление подготовки 09.03.01.05 «Системная и программная инженерия»', { spacing: { before: 120 } }),
    c(meta.work, { bold: true, size: 32, spacing: { before: 2400 } }),
    c('Дисциплина: Методы машинного обучения', { spacing: { before: 240 } }),
    c(`Тема: ${meta.topic}`, { spacing: { before: 120 } }),
    r('Выполнил:', 2000),
    r('студент группы 231-329'),
    r('Арутюнян Ф. Р.'),
    r('Проверил:', 240),
    r('Перепелкина Ю. В.'),
    c('Москва 2026', { spacing: { before: 2400 } }),
    new Paragraph({ pageBreakBefore: true, spacing: { line: 240 }, children: [] }),
  ];
}

function tocBlock(headings, pages) {
  const out = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: 'СОДЕРЖАНИЕ', bold: true })] })];
  headings.forEach((h, i) => {
    // обычный табулятор с точками: PositionalTab LibreOffice рисует без заполнителя
    out.push(new Paragraph({
      spacing: { line: 360 },
      indent: { left: h.level === 2 ? Math.round(5 * MM) : 0 },
      tabStops: [{ type: TabStopType.RIGHT, position: Math.round(TEXT_WIDTH_MM * MM), leader: LeaderType.DOT }],
      children: [new TextRun(h.text), new TextRun({ children: [new Tab(), String(pages ? pages[i] : 999)] })],
    }));
  });
  return out;
}

function parseBody(body, baseDir) {
  const lines = body.split('\n');
  const blocks = [];
  const headings = [];
  let para = [];
  let pendingTableCaption = null;
  let letter = 0;

  const flush = () => {
    if (para.length) blocks.push(bodyPara(para.join(' ')));
    para = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const t = line.trim();

    if (t.startsWith('```')) {
      flush();
      const code = [];
      while (++i < lines.length && !lines[i].trim().startsWith('```')) code.push(lines[i]);
      code.forEach((c, k) => blocks.push(new Paragraph({
        keepNext: k < code.length - 1,
        keepLines: true,
        spacing: { before: k === 0 ? 120 : 0, after: k === code.length - 1 ? 120 : 0 },
        indent: { left: Math.round(5 * MM) },
        children: [new TextRun({ text: c.replace(/\t/g, '    ') || ' ', font: 'Courier New', size: 20 })],
      })));
      continue;
    }
    if (!t) { flush(); letter = 0; continue; }

    let m;
    if ((m = t.match(/^(#{1,2})\s+(.*)$/))) {
      flush();
      const level = m[1].length;
      const text = m[2];
      const structural = text === text.toUpperCase();
      headings.push({ level, text });
      blocks.push(new Paragraph({
        heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
        pageBreakBefore: level === 1,
        keepNext: true,
        alignment: structural ? AlignmentType.CENTER : AlignmentType.LEFT,
        indent: structural ? undefined : { firstLine: Math.round(12.5 * MM) },
        spacing: { before: level === 1 ? 0 : 360, after: 240, line: 360 },
        children: [new TextRun({ text, bold: true, font: FONT, size: 28, color: '000000' })],
      }));
      continue;
    }
    if ((m = t.match(/^!\[(.*)\]\((.*)\)$/))) {
      flush();
      blocks.push(...imageBlock(m[1], path.join(baseDir, m[2])));
      continue;
    }
    if (/^Таблица \d+ — /.test(t) && lines[i + 1] && lines[i + 1].trim().startsWith('|')) {
      flush();
      pendingTableCaption = t;
      continue;
    }
    if (t.startsWith('|')) {
      flush();
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(lines[i++].trim());
      i--;
      blocks.push(...tableBlock(pendingTableCaption, rows));
      pendingTableCaption = null;
      continue;
    }
    if ((m = t.match(/^- (.*)$/))) {
      flush();
      blocks.push(bodyPara(`${LOWER_LETTERS[letter++ % LOWER_LETTERS.length]}) ${m[1]}`));
      continue;
    }
    if ((m = t.match(/^(\d+)\. (.*)$/)) && !/^\d+\.\d/.test(t)) {
      flush();
      // в списке источников по ГОСТ номер с точкой, в перечислениях со скобкой
      const inSources = headings.length && headings[headings.length - 1].text.startsWith('СПИСОК');
      blocks.push(bodyPara(`${m[1]}${inSources ? '.' : ')'} ${m[2]}`));
      continue;
    }
    para.push(t);
  }
  flush();
  return { blocks, headings };
}

function buildDoc(meta, parsed, pages) {
  return new Document({
    styles: {
      default: { document: { run: { font: FONT, size: 28 }, paragraph: { spacing: { line: 360 } } } },
      paragraphStyles: [
        { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 28, bold: true, color: '000000' }, paragraph: { outlineLevel: 0 } },
        { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 28, bold: true, color: '000000' }, paragraph: { outlineLevel: 1 } },
      ],
    },
    sections: [{
      properties: {
        titlePage: true,
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: Math.round(20 * MM), right: Math.round(15 * MM), bottom: Math.round(20 * MM), left: Math.round(30 * MM) },
        },
      },
      footers: {
        first: new Footer({ children: [new Paragraph({ children: [] })] }),
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 28 })] })] }),
      },
      children: [...titlePage(meta), ...tocBlock(parsed.headings, pages), ...parsed.blocks],
    }],
  });
}

function pdfPages(docxPath) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'gost-'));
  execFileSync('soffice', [`-env:UserInstallation=file://${tmp}/profile`, '--headless', '--convert-to', 'pdf', '--outdir', tmp, docxPath], { stdio: 'ignore' });
  const pdf = path.join(tmp, path.basename(docxPath).replace(/\.docx$/, '.pdf'));
  return execFileSync('pdftotext', ['-layout', pdf, '-']).toString().split('\f');
}

const norm = (s) => s.replace(/\s+/g, ' ').trim().toLowerCase();

async function main() {
  const [src, out] = process.argv.slice(2);
  const { meta, body } = parseSource(fs.readFileSync(src, 'utf8'));
  const baseDir = path.dirname(src);

  // первый проход с заглушками вместо номеров страниц
  let parsed = parseBody(body, baseDir);
  fs.writeFileSync(out, await Packer.toBuffer(buildDoc(meta, parsed, null)));

  const pages = pdfPages(out).map(norm);
  // страницы содержания узнаём по заглушке «.....999»
  let from = 0;
  pages.forEach((p, i) => { if (/\.{5,}\s*999/.test(p)) from = i + 1; });
  if (!from) throw new Error('Не нашёл содержание в PDF');
  const found = parsed.headings.map((h) => {
    const key = norm(h.text).slice(0, 40);
    let k = from;
    while (k < pages.length && !pages[k].includes(key)) k++;
    if (k === pages.length) throw new Error(`Заголовок не найден в PDF: ${h.text}`);
    from = k;
    return k + 1;
  });

  parsed = parseBody(body, baseDir);
  fs.writeFileSync(out, await Packer.toBuffer(buildDoc(meta, parsed, found)));
  console.log(`${out}: ${pages.length} стр., заголовков ${found.length}`);
}

main().catch((e) => { console.error(e.message); process.exit(1); });
