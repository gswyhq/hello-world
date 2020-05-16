#! /usr/lib/python3
# -*- coding: utf-8 -*-


from whoosh.fields import Schema, ID, TEXT

schema = Schema(path = ID(unique=True), content=TEXT)

ix = index.create_in("index")
writer = ix.writer()
writer.add_document(path=u"/a", content=u"The first document")
writer.add_document(path=u"/b", content=u"The second document")
writer.commit()

writer = ix.writer()
# Because "path" is marked as unique, calling update_document with path="/a"
# will delete any existing documents where the "path" field contains "/a".
writer.update_document(path=u"/a", content="Replacement for the first document")
writer.commit()


import os.path
from whoosh import index
from whoosh.fields import Schema, ID, TEXT

def clean_index(dirname):
  # Always create the index from scratch
  ix = index.create_in(dirname, schema=get_schema())
  writer = ix.writer()

  # Assume we have a function that gathers the filenames of the
  # documents to be indexed
  for path in my_docs():
    add_doc(writer, path)

  writer.commit()


def get_schema()
  return Schema(path=ID(unique=True, stored=True), content=TEXT)


def add_doc(writer, path):
  fileobj = open(path, "rb")
  content = fileobj.read()
  fileobj.close()
  writer.add_document(path=path, content=content)





def get_schema()
  return Schema(path=ID(unique=True, stored=True), time=STORED, content=TEXT)

def add_doc(writer, path):
  fileobj = open(path, "rb")
  content = fileobj.read()
  fileobj.close()
  modtime = os.path.getmtime(path)
  writer.add_document(path=path, content=content, time=modtime)



def index_my_docs(dirname, clean=False):
  if clean:
    clean_index(dirname)
  else:
    incremental_index(dirname)


def incremental_index(dirname)
    ix = index.open_dir(dirname)

    # The set of all paths in the index
    indexed_paths = set()
    # The set of all paths we need to re-index
    to_index = set()

    with ix.searcher() as searcher:
      writer = ix.writer()

      # Loop over the stored fields in the index
      for fields in searcher.all_stored_fields():
        indexed_path = fields['path']
        indexed_paths.add(indexed_path)

        if not os.path.exists(indexed_path):
          # This file was deleted since it was indexed
          writer.delete_by_term('path', indexed_path)

        else:
          # Check if this file was changed since it
          # was indexed
          indexed_time = fields['time']
          mtime = os.path.getmtime(indexed_path)
          if mtime > indexed_time:
            # The file has changed, delete it and add it to the list of
            # files to reindex
            writer.delete_by_term('path', indexed_path)
            to_index.add(indexed_path)

      # Loop over the files in the filesystem
      # Assume we have a function that gathers the filenames of the
      # documents to be indexed
      for path in my_docs():
        if path in to_index or path not in indexed_paths:
          # This is either a file that's changed, or a new file
          # that wasn't indexed before. So index it!
          add_doc(writer, path)

      writer.commit()


