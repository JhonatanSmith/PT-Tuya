-- Ejercicio 1

SELECT C.NOMBRECIUDAD, COUNT(*) AS CantidadEstudiantes
FROM Estudiantes E -- Mapear Estudiante E
JOIN Ciudad C ON E.CODIGOCIUDAD = C.CODIGOCIUDAD -- Mapear nombre Ciucad a C; luego JOIN Estudiantes y Ciudad por mismma columna
GROUP BY E.CODIGOCIUDAD, C.NOMBRECIUDAD -- Agrupar por ciudad y codigo para contar cantidad estudiantes. 
ORDER BY CantidadEstudiantes DESC; -- Muestrelos ordenados

-- Ejercicio 2: Año de ingreso y carrera

-- ¿QUE SE QUIERE? = Una tabla mas o menos asi: 

-- | AñoIngreso | CARRERA | CantidadEstudiantes|
-- |  2021      | Estadistica |  3 |

-- ¿COMO? 

-- De estudiantes saque la fecha y de la fecha el año (SQL ya da formato)

-- De estudiantes saque tambn carrera y Cuente en la new column CantidadEstudiantes
SELECT EXTRACT(YEAR FROM FECHAINGRESO) AS AnioIngreso, CARRERA, COUNT(*) AS CantidadEstudiantes --  Solo extraer Año de columna FECHA INGRESO
FROM Estudiantes 
-- A este punto ya hay un listado. Agrupe y listo. 
GROUP BY EXTRACT(YEAR FROM FECHAINGRESO), CARRERA
ORDER BY AnioIngreso, CARRERA;

-- Ejercicio 3: Meses transcurridos dse fecha ref

-- Se quier algo asi, esto no se si es a 2025 o la fecha escrita...
SELECT 
  E.ID,
  E.NOMBREAPELLIDO,
  TIMESTAMPDIFF(
    MONTH,
    E.FECHAINGRESO,
    '2023-10-01'
  ) AS MesesTranscurridos
FROM Estudiante AS E;

-- En POstgreSQL, la funcion TIMESTAMPDIFF no existe, pero se puede hacer asi:
SELECT
  E.ID,
  E.NOMBREAPELLIDO,
  (EXTRACT(YEAR FROM AGE('2023-10-01', E.FECHAINGRESO)) * 12
   + EXTRACT(MONTH FROM AGE('2023-10-01', E.FECHAINGRESO))
  )::INT AS MesesTranscurridos
FROM Estudiante E;



-- Ejercicio 4: Ciudaddes en las cuales no hay estudiantes????

SELECT C.NOMBRECIUDAD
FROM Ciudad C
LEFT JOIN Estudiantes E ON C.CODIGOCIUDAD = E.CODIGOCIUDAD
WHERE E.CODIGOCIUDAD IS NULL;

-- Ejercicio 5: Estu sin tel

SELECT *
FROM Estudiantes
WHERE TELEFONO IS NULL OR TELEFONO = '';

-- Ejercicio 6: Cant estu por carrea en la ciudad

SELECT
  C.NOMBRECIUDAD,
  E.CARRERA,
  COUNT(*) AS CantidadEstudiantes
FROM Ciudad C
LEFT JOIN Estudiantes E ON C.CODIGOCIUDAD = E.CODIGOCIUDAD
GROUP BY C.NOMBRECIUDAD, E.CARRERA
ORDER BY C.NOMBRECIUDAD, CantidadEstudiantes DESC;

-- Ejercicio 7: verifique @ en correo

-- Hago un update? Supongo que si
UPDATE Estudiantes
SET CORREO = NULL
WHERE POSITION('@' IN CORREO) = 0 OR CORREO IS NULL;
