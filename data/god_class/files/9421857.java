public class DTMDOMException extends org.w3c.dom.DOMException
{
    static final long serialVersionUID = 1895654266613192414L;
  /**
   * Constructs a DOM/DTM exception.
   *
   * @param code
   * @param message
   */
  public DTMDOMException(short code, String message)
  {
    super(code, message);
  }

  /**
   * Constructor DTMDOMException
   *
   *
   * @param code
   */
  public DTMDOMException(short code)
  {
    super(code, "");
  }
}