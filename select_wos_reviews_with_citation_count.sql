USE wos;

SELECT a.UID, a.doi, a.pub_date, a.title, a.title_source, a.title_source_abbrev, a.doc_type, COUNT(b.UID) AS num_citations
FROM papers AS a
LEFT JOIN citations AS b ON a.UID = b.UID
WHERE a.doc_type = 'Review; Review'
GROUP BY a.UID, a.doi, a.pub_date, a.title, a.title_source, a.title_source_abbrev, a.doc_type;
