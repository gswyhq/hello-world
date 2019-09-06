| Cypher Type      | Parameter | Result |
| ---------------- | --------- | ------ |
| `null`*          | ✔         | ✔      |
| `List`           | ✔         | ✔      |
| `Map`            | ✔         | ✔      |
| `Boolean`        | ✔         | ✔      |
| `Integer`        | ✔         | ✔      |
| `Float`          | ✔         | ✔      |
| `String`         | ✔         | ✔      |
| `ByteArray`      | ✔         | ✔      |
| `Date`           | ✔         | ✔      |
| `Time`           | ✔         | ✔      |
| `LocalTime`      | ✔         | ✔      |
| `DateTime`       | ✔         | ✔      |
| `LocalDateTime`  | ✔         | ✔      |
| `Duration`       | ✔         | ✔      |
| `Point`          | ✔         | ✔      |
| `Node`**         |           | ✔      |
| `Relationship`** |           | ✔      |
| `Path`**         |           | ✔      |



| Neo4j type      | Python 2 type           | Python 3 type           |
| --------------- | ----------------------- | ----------------------- |
| `null`          | `None`                  | `None`                  |
| `List`          | `list`                  | `list`                  |
| `Map`           | `dict`                  | `dict`                  |
| `Boolean`       | `bool`                  | `bool`                  |
| `Integer`       | `int / long`*           | `int`                   |
| `Float`         | `float`                 | `float`                 |
| `String`        | `unicode`**             | `str`                   |
| `ByteArray`     | `bytearray`             | `bytearray`             |
| `Date`          | **neotime.Date**        | **neotime.Date**        |
| `Time`          | **neotime.Time**†       | **neotime.Time**†       |
| `LocalTime`     | **neotime.Time**††      | **neotime.Time**††      |
| `DateTime`      | **neotime.DateTime**†   | **neotime.DateTime**†   |
| `LocalDateTime` | **neotime.DateTime**††  | **neotime.DateTime**††  |
| `Duration`      | **neotime.Duration***** | **neotime.Duration***** |
| `Point`         | **Point**               | **Point**               |
| `Node`          | **Node**                | **Node**                |
| `Relationship`  | **Relationship**        | **Relationship**        |
| `Path`          | **Path**                | **Path**                |



* https://neo4j.com/docs/driver-manual/current/cypher-values/

